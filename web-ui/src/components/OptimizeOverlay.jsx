import { useRef, useEffect } from 'react';

const ABBREV = {
  collapse_rate: 'CR',
  lj_strength: 'LJ',
  perturbation: 'P',
  epsilon: 'E',
  damping: 'D',
};

function formatParams(params) {
  if (!params) return '\u2014';
  return Object.entries(params)
    .map(([k, v]) => `${ABBREV[k] || k}:${v.toFixed(2)}`)
    .join('  ');
}

export default function OptimizeOverlay({ optTrials, bestOptResult, onStop, theme }) {
  const canvasRef = useRef(null);
  const latest = optTrials[optTrials.length - 1];
  const trial = latest?.trial || 0;
  const maxTrials = latest?.max_trials || 60;
  const t = theme;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || optTrials.length === 0) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const maxT = optTrials[0]?.max_trials || 60;
    const distances = optTrials.map(t => t.distance);
    const minD = Math.min(...distances);
    const maxD = Math.max(...distances);
    const pad = (maxD - minD) * 0.1 || 1;
    const lo = minD - pad, hi = maxD + pad;

    const x = (i) => (i / maxT) * W;
    const y = (d) => H - ((d - lo) / (hi - lo)) * H;

    // Optimal reference line (dashed)
    const optCost = optTrials[0]?.optimal_cost;
    if (optCost && optCost >= lo && optCost <= hi) {
      ctx.strokeStyle = t.good;
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(0, y(optCost));
      ctx.lineTo(W, y(optCost));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Trial dots (accent color, 40% opacity)
    const accentRgb = hexToRgb(t.accent);
    ctx.fillStyle = accentRgb ? `rgba(${accentRgb},0.4)` : 'rgba(192,88,42,0.4)';
    optTrials.forEach((tr, i) => {
      ctx.beginPath();
      ctx.arc(x(i), y(tr.distance), 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // Best-so-far line (good color)
    ctx.strokeStyle = t.good;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let best = Infinity;
    optTrials.forEach((tr, i) => {
      best = Math.min(best, tr.distance);
      const px = x(i), py = y(best);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }, [optTrials, t]);

  const bestDist = bestOptResult?.best_distance ?? latest?.best_distance;
  const bestParams = bestOptResult?.best_params ?? latest?.best_params;
  const bestGapPct = latest?.best_gap_pct;

  return (
    <div style={{
      position: 'fixed',
      top: 16,
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 40,
      background: t.panelBg + 'f0',
      backdropFilter: 'blur(12px)',
      border: `1px solid ${t.panelBorder}`,
      borderRadius: 10,
      width: 440,
      padding: '16px 20px',
      fontFamily: 'monospace',
      fontSize: 10,
      color: t.text,
    }}>
      {/* Header */}
      <div style={{
        fontSize: 9,
        textTransform: 'uppercase',
        letterSpacing: 2,
        color: t.accent,
        textAlign: 'center',
        marginBottom: 12,
        fontWeight: 600,
      }}>
        Bayesian Optimization
      </div>

      {/* Trial counter */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ color: t.textDim }}>TRIAL</span>
        <span style={{ color: t.textBright, fontWeight: 600 }}>{trial} / {maxTrials}</span>
      </div>

      {/* Current params */}
      <div style={{ color: t.accent, marginBottom: 4, fontSize: 9 }}>
        {formatParams(latest?.params)}
      </div>

      {/* Current distance */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span style={{ color: t.textDim }}>Current</span>
        <span style={{ color: t.accent }}>
          {latest?.distance != null ? latest.distance.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '\u2014'}
        </span>
      </div>

      {/* Best distance + gap */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
        <span style={{ color: t.textDim }}>Best</span>
        <span style={{ color: t.good, fontWeight: 600 }}>
          {bestDist != null ? bestDist.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '\u2014'}
          {bestGapPct != null && (
            <span style={{ marginLeft: 6 }}>({bestGapPct.toFixed(1)}%)</span>
          )}
        </span>
      </div>

      {/* Convergence chart */}
      <div style={{
        background: t.selectBg,
        border: `1px solid ${t.panelBorder}`,
        borderRadius: 6,
        overflow: 'hidden',
        marginBottom: 10,
      }}>
        <canvas
          ref={canvasRef}
          width={800}
          height={200}
          style={{ width: '100%', height: 100, display: 'block' }}
        />
      </div>

      {/* Best params */}
      <div style={{ color: t.good, marginBottom: 12, fontSize: 9 }}>
        Best: {formatParams(bestParams)}
      </div>

      {/* Stop button */}
      <button
        onClick={onStop}
        style={{
          width: '100%',
          padding: '8px 0',
          background: t.accent,
          border: 'none',
          borderRadius: 6,
          color: '#fff',
          fontFamily: 'monospace',
          fontSize: 10,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: 1,
          cursor: 'pointer',
        }}
      >
        Stop &amp; Use Best
      </button>
    </div>
  );
}

function hexToRgb(hex) {
  if (!hex || hex[0] !== '#') return null;
  const n = parseInt(hex.slice(1), 16);
  return `${(n >> 16) & 255},${(n >> 8) & 255},${n & 255}`;
}
