import { useState } from 'react';

const PARAM_PRESETS = {
  torus: [
    { name: 'collapse_rate', label: 'Collapse Rate', bounds: [0.02, 0.45] },
    { name: 'lj_strength', label: 'LJ Strength', bounds: [0.2, 4.5] },
    { name: 'perturbation', label: 'Perturbation', bounds: [0.0, 1.0] },
    { name: 'epsilon', label: 'Epsilon', bounds: [0.02, 0.25] },
  ],
  annular: [
    { name: 'collapse_rate', label: 'Collapse Rate', bounds: [0.02, 0.45] },
    { name: 'lj_strength', label: 'LJ Strength', bounds: [0.2, 4.5] },
    { name: 'epsilon', label: 'Epsilon', bounds: [0.02, 0.25] },
  ],
  '2d': [
    { name: 'collapse_rate', label: 'Collapse Rate', bounds: [0.001, 0.05] },
    { name: 'lj_strength', label: 'LJ Strength', bounds: [0.1, 50.0] },
    { name: 'damping', label: 'Damping', bounds: [0.5, 1.0] },
  ],
};

export default function OptimizeModal({ mode, theme, onStart, onClose }) {
  const presets = PARAM_PRESETS[mode] || PARAM_PRESETS.torus;
  const [selected, setSelected] = useState(() => presets.map(() => true));
  const [maxTrials, setMaxTrials] = useState(60);
  const [optMode, setOptMode] = useState('fast');

  const toggleParam = (i) => {
    setSelected(s => s.map((v, j) => j === i ? !v : v));
  };

  const handleStart = () => {
    const params = [];
    const bounds = [];
    presets.forEach((p, i) => {
      if (selected[i]) {
        params.push(p.name);
        bounds.push(p.bounds);
      }
    });
    if (params.length === 0) return;
    onStart({ params, bounds, max_trials: maxTrials, mode: optMode });
  };

  const t = theme;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.7)',
        zIndex: 50,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        style={{
          background: '#1c1e24',
          border: '1px solid #3a3d48',
          borderRadius: 12,
          padding: '20px 24px',
          maxWidth: 400,
          width: '100%',
          boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
          fontFamily: "'JetBrains Mono', monospace",
        }}
      >
        {/* Title */}
        <h3
          className="font-mono text-sm font-semibold tracking-wider uppercase"
          style={{ color: t.accent, marginBottom: 16 }}
        >
          Bayesian Optimization
        </h3>

        {/* Parameter checkboxes */}
        <div style={{ marginBottom: 16 }}>
          <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: t.textDim, marginBottom: 8 }}>
            Parameters
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {presets.map((p, i) => (
              <label
                key={p.name}
                style={{
                  background: '#14161a',
                  border: '1px solid #2a2d36',
                  borderRadius: 6,
                  padding: '6px 8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  cursor: 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={selected[i]}
                  onChange={() => toggleParam(i)}
                  style={{ accentColor: t.accent }}
                />
                <span className="font-mono text-xs" style={{ color: t.textBright, flex: 1 }}>
                  {p.label}
                </span>
                <span className="font-mono text-[10px]" style={{ color: t.textDim }}>
                  [{p.bounds[0]}, {p.bounds[1]}]
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Trial count */}
        <div style={{ marginBottom: 16 }}>
          <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: t.textDim, marginBottom: 6 }}>
            Max Trials
          </div>
          <input
            type="number"
            min={1}
            max={500}
            value={maxTrials}
            onChange={e => setMaxTrials(Math.max(1, parseInt(e.target.value) || 1))}
            className="font-mono text-xs"
            style={{
              background: '#14161a',
              border: '1px solid #2a2d36',
              borderRadius: 6,
              padding: '6px 10px',
              color: t.textBright,
              width: 80,
              outline: 'none',
            }}
          />
        </div>

        {/* Fast / Visual toggle */}
        <div style={{ marginBottom: 20 }}>
          <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: t.textDim, marginBottom: 6 }}>
            Mode
          </div>
          <div style={{ display: 'flex', borderRadius: 6, overflow: 'hidden', border: '1px solid #2a2d36' }}>
            {['fast', 'visual'].map(m => (
              <button
                key={m}
                onClick={() => setOptMode(m)}
                className="font-mono text-xs uppercase tracking-wider"
                style={{
                  flex: 1,
                  padding: '6px 0',
                  border: 'none',
                  cursor: 'pointer',
                  background: optMode === m ? t.accent : '#14161a',
                  color: optMode === m ? '#fff' : t.textDim,
                  fontWeight: optMode === m ? 600 : 400,
                  transition: 'background 0.15s, color 0.15s',
                }}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            className="font-mono text-xs uppercase tracking-wider"
            style={{
              padding: '7px 18px',
              borderRadius: 6,
              border: '1px solid #3a3d48',
              background: 'transparent',
              color: t.textDim,
              cursor: 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleStart}
            className="font-mono text-xs uppercase tracking-wider font-semibold"
            style={{
              padding: '7px 18px',
              borderRadius: 6,
              border: 'none',
              background: t.accent,
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Start
          </button>
        </div>
      </div>
    </div>
  );
}
