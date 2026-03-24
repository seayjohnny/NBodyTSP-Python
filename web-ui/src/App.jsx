import { useState, useCallback } from 'react';
import SetupScreen from './components/SetupScreen';
import ControlPanel from './components/ControlPanel';
import SimCanvas from './components/SimCanvas';
import FullTourMap from './components/FullTourMap';
import OptimizeModal from './components/OptimizeModal';
import OptimizeOverlay from './components/OptimizeOverlay';
import { useSimulation } from './hooks/useSimulation';
import { getTheme, THEME_NAMES } from './themes';

export default function App() {
  const [screen, setScreen] = useState('setup');
  const [mode, setMode] = useState('torus');
  const [themeName, setThemeName] = useState('dark');
  const [theme, setTheme] = useState(() => getTheme('dark'));
  const [view, setView] = useState('sim');
  const [showOptModal, setShowOptModal] = useState(false);
  const {
    connect, disconnect, sendCmd, sendParam,
    sendOptStart, sendOptStop,
    metadata, stateRef,
    optimizing, optTrials, bestOptResult,
  } = useSimulation();

  const handleThemeChange = useCallback((name) => {
    setThemeName(name);
    setTheme(getTheme(name));
  }, []);

  const handleStart = useCallback((config) => {
    setMode(config.mode);
    handleThemeChange(config.theme || 'dark');
    setScreen('sim');
    setView('sim');
    connect(config, () => {});
  }, [connect, handleThemeChange]);

  const handleMenu = useCallback(() => {
    disconnect();
    setScreen('setup');
    setView('sim');
  }, [disconnect]);

  const handleOptStart = useCallback((config) => {
    setShowOptModal(false);
    sendOptStart(config);
  }, [sendOptStart]);

  if (screen === 'setup') {
    return (
      <div className="w-full h-full bg-[#0f0f12]">
        <SetupScreen onStart={handleStart} />
      </div>
    );
  }

  const hasTour = metadata?.tour_positions != null;
  const bgStyle = { background: `rgb(${theme.bg.map(c => Math.round(c * 255)).join(',')})` };

  return (
    <div className="w-full h-full flex" style={bgStyle}>
      <div className="flex-1 relative">
        {/* View toggle tabs */}
        {hasTour && (
          <div className="absolute top-3 left-3 z-10 flex rounded-lg overflow-hidden border border-white/10 shadow-lg"
               style={{ background: theme.panelBg }}>
            <button
              onClick={() => setView('sim')}
              className={`px-4 py-1.5 text-[11px] font-mono font-semibold tracking-wider transition-colors cursor-pointer
                ${view === 'sim' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`}
              style={view === 'sim' ? { background: theme.accent } : {}}>
              SIMULATION
            </button>
            <button
              onClick={() => setView('tour')}
              className={`px-4 py-1.5 text-[11px] font-mono font-semibold tracking-wider transition-colors cursor-pointer
                ${view === 'tour' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`}
              style={view === 'tour' ? { background: theme.accent } : {}}>
              TOUR MAP
            </button>
          </div>
        )}

        {/* Main view area */}
        <div className={view === 'sim' ? 'w-full h-full' : 'hidden'}>
          <SimCanvas stateRef={stateRef} mode={mode} theme={theme} metadata={metadata} />
        </div>
        {view === 'tour' && hasTour && (
          <FullTourMap metadata={metadata} theme={theme} />
        )}
      </div>
      <ControlPanel
        mode={mode}
        theme={theme}
        themeName={themeName}
        onThemeChange={handleThemeChange}
        sendParam={sendParam}
        sendCmd={sendCmd}
        metadata={metadata}
        onMenu={handleMenu}
        onOptimize={() => setShowOptModal(true)}
        optimizing={optimizing}
        bestOptResult={bestOptResult}
      />
      {showOptModal && (
        <OptimizeModal
          mode={mode}
          theme={theme}
          onStart={handleOptStart}
          onClose={() => setShowOptModal(false)}
        />
      )}
      {optimizing && (
        <OptimizeOverlay
          optTrials={optTrials}
          bestOptResult={bestOptResult}
          onStop={sendOptStop}
          theme={theme}
        />
      )}
    </div>
  );
}
