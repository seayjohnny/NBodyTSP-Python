import { useRef, useState, useCallback, useEffect } from 'react';

export function useSimulation() {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [metadata, setMetadata] = useState({});
  const [optimizing, setOptimizing] = useState(false);
  const [optTrials, setOptTrials] = useState([]);
  const [bestOptResult, setBestOptResult] = useState(null);

  const stateRef = useRef({
    positions: null, velocities: null, N: 0,
    R: 0, r: 0, r0: 0, phase: 0,
    innerR: 0, outerR: 1, mode: 'torus',
  });
  const frameCallbackRef = useRef(null);

  const connect = useCallback((config, onFrame) => {
    frameCallbackRef.current = onFrame;
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${location.host}/ws`);
    wsRef.current = ws;
    stateRef.current.mode = config.mode;

    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      setConnected(true);
      ws.send(JSON.stringify({ cmd: 'init', config }));
    };
    ws.onmessage = (e) => {
      if (typeof e.data === 'string') {
        const msg = JSON.parse(e.data);
        if (msg.type === 'metadata') setMetadata(msg.data);
        else if (msg.type === 'opt_trial') {
          setOptTrials(prev => [...prev, msg.data]);
          setBestOptResult(msg.data);
        }
        else if (msg.type === 'opt_complete') {
          setOptimizing(false);
          setBestOptResult(msg.data);
        }
      } else {
        parseState(e.data, stateRef.current);
        if (frameCallbackRef.current) frameCallbackRef.current(stateRef.current);
      }
    };
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    setConnected(false);
    setMetadata({});
    setOptimizing(false);
    setOptTrials([]);
    setBestOptResult(null);
  }, []);

  const sendCmd = useCallback((cmd) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify({ cmd }));
  }, []);

  const sendParam = useCallback((key, value) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify({ cmd: 'set_param', key, value }));
  }, []);

  const sendOptStart = useCallback((config) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      setOptimizing(true);
      setOptTrials([]);
      setBestOptResult(null);
      wsRef.current.send(JSON.stringify({ cmd: 'optimize_start', config }));
    }
  }, []);

  const sendOptStop = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify({ cmd: 'optimize_stop' }));
  }, []);

  useEffect(() => () => disconnect(), [disconnect]);

  return {
    connect, disconnect, sendCmd, sendParam,
    sendOptStart, sendOptStop,
    connected, metadata, stateRef,
    optimizing, optTrials, bestOptResult,
  };
}

function parseState(ab, state) {
  const buf = new DataView(ab);
  const mode = buf.getUint32(0, true);

  if (mode === 0) {
    state.mode = 'torus';
    state.N = buf.getUint32(4, true);
    state.R = buf.getFloat32(8, true);
    state.r = buf.getFloat32(12, true);
    state.r0 = buf.getFloat32(16, true);
    state.phase = buf.getUint32(20, true);
    const h = 24, n3 = state.N * 3;
    state.positions = new Float32Array(ab, h, n3);
    state.velocities = new Float32Array(ab, h + n3 * 4, n3);
  } else {
    state.mode = (mode === 2) ? 'annular' : '2d';
    state.N = buf.getUint32(4, true);
    state.innerR = buf.getFloat32(8, true);
    state.outerR = buf.getFloat32(12, true);
    state.phase = buf.getUint32(16, true);
    const h = 20, n2 = state.N * 2;
    state.positions = new Float32Array(ab, h, n2);
    state.velocities = new Float32Array(ab, h + n2 * 4, n2);
  }
}
