import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { HotkeyProvider } from './components/controls/Hotkeys';
import { WorkspaceLayoutProvider } from './components/workstation/WorkspaceLayouts';
import '@fontsource-variable/inter'; // self-hosted: bundled into dist, no font CDN
import './styles.css';
import './stage19.css';
import { installResourceLedgerIfRequested } from './utils/resourceLedger';
installResourceLedgerIfRequested(); // Stage 18: opt-in (?r3d-diag=1) object-URL / WebSocket / WebGL-context accounting
createRoot(document.getElementById('root')!).render(<StrictMode><HotkeyProvider><WorkspaceLayoutProvider><App /></WorkspaceLayoutProvider></HotkeyProvider></StrictMode>);
