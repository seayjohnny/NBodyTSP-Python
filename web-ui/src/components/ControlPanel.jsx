import { useState, useEffect, useRef } from 'react';
import TourMap from './TourMap';
import { THEMES, THEME_NAMES } from '../themes';

function Slider({ label, value, min, max, step, fmt, onChange, theme }) {
  const display = fmt === 'd' ? Math.round(value) : value.toFixed(2);
  return (
    <div>
      <div className="flex justify-between text-[10px] font-mono uppercase tracking-wider" style={{ color: theme.textDim }}>
        <span>{label}</span>
        <span style={{ color: theme.textBright }}>{display}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full"
        style={{ background: theme.sliderTrack, accentColor: theme.accent }} />
    </div>
  );
}

export default function ControlPanel({ mode, theme, themeName, onThemeChange, sendParam, sendCmd, metadata, onMenu }) {
  const isTorus = mode === 'torus' || mode === 'annular';
  const [params, setParams] = useState(
    isTorus
      ? { shrink_rate: 0.10, epsilon: 0.08, lj_strength: 1.0, perturbation: 0.50, substeps: 4 }
      : { wall_strength: 20000, damp: 20, dt: 0.01, dr: 0.01, slope_repulsion: 50, mag_attraction: 25, lj_strength: 1.0, substeps: 4 }
  );

  const synced = useRef(false);
  useEffect(() => {
    if (!isTorus && metadata?.wall_strength != null && !synced.current) {
      setParams(p => ({
        ...p,
        wall_strength: metadata.wall_strength,
        damp: metadata.damp,
        dt: metadata.dt,
        dr: metadata.dr,
        lj_strength: metadata.lj_strength ?? p.lj_strength,
      }));
      synced.current = true;
    }
  }, [metadata, isTorus]);

  const setP = (key, value) => {
    setParams(p => ({ ...p, [key]: value }));
    sendParam(key, value);
  };

  const phase = metadata.phase || 'READY';
  const phaseColor = (phase === 'COLLAPSING' || phase === 'RUNNING') ? theme.warn
    : (phase === 'CIRCLE' || phase === 'COLLAPSED') ? theme.accent : theme.good;

  const t = theme; // shorthand

  const selectStyle = {
    background: t.selectBg, border: `1px solid ${t.panelBorder}`,
    color: t.textBright, borderRadius: '0.375rem', padding: '0.25rem 0.5rem',
    fontSize: '0.75rem', fontFamily: 'monospace', outline: 'none', width: '100%',
    cursor: 'pointer', WebkitAppearance: 'none', appearance: 'none',
  };

  return (
    <div className="w-72 border-l flex flex-col overflow-y-auto h-full"
         style={{ background: t.panelBg, borderColor: t.panelBorder }}>

      {/* Title */}
      <div className="p-4 border-b" style={{ borderColor: t.panelBorder }}>
        <h2 className="font-mono text-xs font-semibold tracking-wider uppercase" style={{ color: t.accent }}>
          {mode === 'torus' ? 'Torus Controls' : mode === 'annular' ? 'Annular Controls' : '2D Controls'}
        </h2>
      </div>

      {/* Sliders */}
      <div className="p-4 space-y-3 flex-1">
        {isTorus ? (
          <>
            <Slider label="Shrink Rate" value={params.shrink_rate} min={0.01} max={0.5} step={0.01} onChange={v => setP('shrink_rate', v)} theme={t} />
            <Slider label="Epsilon" value={params.epsilon} min={0.01} max={0.3} step={0.01} onChange={v => setP('epsilon', v)} theme={t} />
            <Slider label="LJ Strength" value={params.lj_strength} min={0.1} max={5} step={0.1} onChange={v => setP('lj_strength', v)} theme={t} />
            {mode === 'torus' && <Slider label="Perturbation" value={params.perturbation} min={0} max={1} step={0.05} onChange={v => setP('perturbation', v)} theme={t} />}
            <Slider label="Speed" value={params.substeps} min={1} max={64} step={1} fmt="d" onChange={v => setP('substeps', v)} theme={t} />
          </>
        ) : (
          <>
            <Slider label="Wall Strength" value={params.wall_strength} min={100} max={50000} step={100} fmt="d" onChange={v => setP('wall_strength', v)} theme={t} />
            <Slider label="Damping" value={params.damp} min={1} max={100} step={1} onChange={v => setP('damp', v)} theme={t} />
            <Slider label="LJ Strength" value={params.lj_strength} min={0.1} max={5} step={0.1} onChange={v => setP('lj_strength', v)} theme={t} />
            <Slider label="DT" value={params.dt} min={0.001} max={0.05} step={0.001} onChange={v => setP('dt', v)} theme={t} />
            <Slider label="DR" value={params.dr} min={0.001} max={0.05} step={0.001} onChange={v => setP('dr', v)} theme={t} />
            <Slider label="Speed" value={params.substeps} min={1} max={64} step={1} fmt="d" onChange={v => setP('substeps', v)} theme={t} />
          </>
        )}

        {/* Buttons */}
        <div className="flex gap-2 pt-2">
          <button onClick={() => sendCmd('start')}
            className="flex-1 text-white text-xs font-mono font-semibold py-2 rounded-md cursor-pointer"
            style={{ background: t.accent }}>
            {isTorus ? 'COLLAPSE' : 'START'}
          </button>
          <button onClick={() => sendCmd('reset')}
            className="flex-1 text-xs font-mono font-semibold py-2 rounded-md border cursor-pointer"
            style={{ background: t.panelLight, borderColor: t.panelBorder, color: t.text }}>
            RESET
          </button>
        </div>

        {mode === 'torus' && (
          <div className="pt-1">
            <label className="block text-[10px] font-mono uppercase tracking-wider mb-1" style={{ color: t.textDim }}>Embedding</label>
            <select onChange={e => sendParam('embed_mode', e.target.value)} defaultValue="flat" style={selectStyle}>
              <option value="flat">Flat Plane</option>
              <option value="centroid">Centroid Angle</option>
              <option value="linear">Linear X</option>
            </select>
          </div>
        )}

        {/* Local search */}
        <div className="pt-2 space-y-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={params.use_local_search || false}
              onChange={e => { setParams(p => ({ ...p, use_local_search: e.target.checked })); sendParam('use_local_search', e.target.checked); }}
              style={{ accentColor: t.accent }} className="w-3.5 h-3.5" />
            <span className="text-[10px] font-mono uppercase tracking-wider" style={{ color: t.textDim }}>Local Search</span>
          </label>
          <select onChange={e => { setParams(p => ({ ...p, local_search_mode: e.target.value })); sendParam('local_search_mode', e.target.value); }}
            defaultValue="2-opt" style={selectStyle}>
            <option value="2-opt">2-opt</option>
            <option value="3-opt">3-opt</option>
            <option value="both">2-opt + 3-opt</option>
          </select>
        </div>

        {/* Theme selector */}
        <div className="pt-2">
          <label className="block text-[10px] font-mono uppercase tracking-wider mb-1" style={{ color: t.textDim }}>Theme</label>
          <select value={themeName} onChange={e => onThemeChange(e.target.value)} style={selectStyle}>
            {THEME_NAMES.map(name => (
              <option key={name} value={name}>{THEMES[name].name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Results */}
      {metadata.tour_distance != null && (
        <div className="border-t p-4 space-y-1.5" style={{ borderColor: t.panelBorder }}>
          <h3 className="font-mono text-[10px] font-semibold uppercase tracking-wider" style={{ color: t.textDim }}>Results</h3>
          <Row label="Tour Distance" value={Math.round(metadata.tour_distance).toLocaleString()} color={t.accent} theme={t} />
          {metadata.optimal_cost != null && <Row label="Optimal" value={Math.round(metadata.optimal_cost).toLocaleString()} color={t.good} theme={t} />}
          {metadata.gap_pct != null && (
            <Row label="Gap" value={metadata.gap_pct.toFixed(2) + '%'}
              color={metadata.gap_pct < 10 ? t.good : metadata.gap_pct < 30 ? t.warn : t.bad} theme={t} />
          )}
          {metadata.nn_cost != null && (
            <>
              <Row label="Nearest Neighbor" value={Math.round(metadata.nn_cost).toLocaleString()} color={t.text} theme={t} />
              <Row label="vs NN"
                value={(metadata.nn_improvement_pct >= 0 ? '+' : '') + metadata.nn_improvement_pct.toFixed(1) + '%'}
                color={metadata.nn_improvement_pct >= 0 ? t.good : t.bad} theme={t} />
            </>
          )}
          {metadata.ls_method && (
            <div className="pt-1 border-t" style={{ borderColor: t.panelBorder }}>
              <Row label="Local Search" value={metadata.ls_method} color={t.accent} theme={t} />
              <Row label="Before" value={Math.round(metadata.ls_before).toLocaleString()} color={t.text} theme={t} />
              <Row label="After" value={Math.round(metadata.ls_after).toLocaleString()} color={t.good} theme={t} />
              <Row label="Improved" value={metadata.ls_pct.toFixed(1) + '%'} color={t.good} theme={t} />
            </div>
          )}
        </div>
      )}

      {/* Tour map */}
      <TourMap metadata={metadata} theme={theme} />

      {/* Status */}
      <div className="border-t p-4 space-y-1" style={{ borderColor: t.panelBorder }}>
        <div className="text-xs font-mono font-semibold" style={{ color: phaseColor }}>
          Phase: {phase}
        </div>
        {metadata.R !== undefined && (
          <>
            <Row label="Tube R" value={metadata.r?.toFixed(3)} color={t.text} theme={t} />
            <Row label="Major R" value={metadata.R?.toFixed(3)} color={t.text} theme={t} />
          </>
        )}
        {metadata.inner_radius !== undefined && (
          <>
            <Row label="Inner R" value={metadata.inner_radius?.toFixed(3)} color={t.text} theme={t} />
            <Row label="Outer R" value={metadata.outer_radius?.toFixed(3)} color={t.text} theme={t} />
          </>
        )}
      </div>

      {/* Menu */}
      <div className="border-t p-4" style={{ borderColor: t.panelBorder }}>
        <button onClick={onMenu}
          className="w-full text-xs font-mono py-2 rounded-md border cursor-pointer"
          style={{ background: t.panelLight, borderColor: t.panelBorder, color: t.textDim }}>
          MENU
        </button>
      </div>
    </div>
  );
}

function Row({ label, value, color, theme }) {
  return (
    <div className="flex justify-between text-[10px] font-mono" style={{ color: theme.textDim }}>
      <span>{label}</span>
      <span className="font-semibold" style={{ color }}>{value ?? '—'}</span>
    </div>
  );
}
