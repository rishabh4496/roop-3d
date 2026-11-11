import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import WorkstationErrorBoundary from './components/common/WorkstationErrorBoundary';
import { HotkeyProvider } from './components/controls/Hotkeys';
import { WorkspaceLayoutProvider } from './components/workstation/WorkspaceLayouts';
import '@fontsource-variable/inter'; // self-hosted: bundled into dist, no font CDN
import './index.css'; // Obsidian Studio tokens (ui-layouts foundation): new names only, loaded before the older layers
import './styles.css';
import './stage19.css';
import './shell/shell.css'; // redesign layer: layers over the older stylesheets
import './components/ui-layouts/ui-layouts.css'; // ui-layouts primitives
import './shell/stage20.css'; // Stage 20: telemetry header, pipeline mini-map, magnetic dock
import './stage23.css'; // Stage 23: actors, mask frame ranges, 3D mesh canvas mode
import './workstation3d.css'; // 3D workstation layer (flow bar, coverage radar, morph/light controls, mesh overlay): must stay last
import { installResourceLedgerIfRequested } from './utils/resourceLedger';
import { installGlobalErrorHandlers } from './utils/globalErrors';
installResourceLedgerIfRequested(); // Stage 18: opt-in (?r3d-diag=1) object-URL / WebSocket / WebGL-context accounting
installGlobalErrorHandlers();       // Stage 13: unhandled rejections become one toast + one console warning
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <WorkstationErrorBoundary name="The workstation" variant="app">
      <HotkeyProvider><WorkspaceLayoutProvider><App /></WorkspaceLayoutProvider></HotkeyProvider>
    </WorkstationErrorBoundary>
  </StrictMode>,
);
