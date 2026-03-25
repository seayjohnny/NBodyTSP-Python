import { useState, useEffect } from 'react';
import { THEMES, THEME_NAMES } from '../themes';

export default function SetupScreen({ onStart }) {
  const [datasets, setDatasets] = useState([]);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [config, setConfig] = useState({
    dataset_path: '', mode: 'torus', wall_force_mode: 'inverse_square', force_mode: 'true_lj',
    theme: 'dark', use_gpu: true,
  });

  useEffect(() => {
    fetch('/api/datasets').then(r => r.json()).then(ds => {
      setDatasets(ds);
      if (ds.length > 0) setConfig(c => ({ ...c, dataset_path: ds[0].path }));
    });
    fetch('/api/info').then(r => r.json()).then(info => {
      setGpuAvailable(info.gpu_available);
    }).catch(() => {});
  }, []);

  const set = (key) => (e) => setConfig(c => ({ ...c, [key]: e.target.value }));

  return (
    <div className="flex items-center justify-center w-full h-full">
      <div className="bg-[#1c1e24] rounded-xl border border-[#2e3038] p-8 w-[440px] shadow-2xl">
        <h1 className="text-2xl font-bold text-white mb-1">N-Body TSP Simulator</h1>
        <p className="text-xs text-gray-500 font-mono mb-6">
          Lennard-Jones physics for the Traveling Salesman Problem
        </p>

        <div className="space-y-4">
          <Field label="Dataset">
            <select value={config.dataset_path} onChange={set('dataset_path')} className="select-field">
              {datasets.map(d => (
                <option key={d.path} value={d.path}>
                  {d.name} ({d.n_cities} cities){d.has_optimal ? ' *' : ''}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Mode">
            <select value={config.mode} onChange={set('mode')} className="select-field">
              <option value="torus">Torus 3D</option>
              <option value="annular">2D Annular</option>
              <option value="2d">2D Walls (legacy)</option>
            </select>
          </Field>

          <Field label="Particle Force Model">
            <select value={config.force_mode} onChange={set('force_mode')} className="select-field">
              <option value="piecewise">Piecewise LJ</option>
              <option value="smooth">Smooth LJ</option>
              <option value="true_lj">True LJ (12-6)</option>
            </select>
          </Field>
          <Field label="Wall Force Model">
            <select value={config.wall_force_mode} onChange={set('wall_force_mode')} className="select-field">
              <option value="linear">Linear Spring</option>
              <option value="inverse_square">Inverse Square</option>
            </select>
          </Field>

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={config.use_gpu}
                onChange={e => setConfig(c => ({ ...c, use_gpu: e.target.checked }))}
                disabled={!gpuAvailable}
                defaultChecked={true}
                className="accent-[#c0582a] w-4 h-4"
              />
              <span className={`text-xs font-mono uppercase tracking-wider ${gpuAvailable ? 'text-gray-300' : 'text-gray-600'}`}>
                GPU Compute {!gpuAvailable && '(unavailable)'}
              </span>
            </label>
          </div>

          <button
            onClick={() => onStart(config)}
            className="w-full bg-[#c0582a] hover:bg-[#dc6e3c] text-white font-semibold py-3 rounded-lg transition-colors mt-4 cursor-pointer"
          >
            Start Simulation
          </button>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div>
      <label className="block text-[10px] font-mono uppercase tracking-wider text-gray-500 mb-1">
        {label}
      </label>
      {children}
    </div>
  );
}
