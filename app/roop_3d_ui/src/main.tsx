import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { HotkeyProvider } from './components/controls/Hotkeys';
import { WorkspaceLayoutProvider } from './components/workstation/WorkspaceLayouts';
import './styles.css';
createRoot(document.getElementById('root')!).render(<StrictMode><HotkeyProvider><WorkspaceLayoutProvider><App /></WorkspaceLayoutProvider></HotkeyProvider></StrictMode>);
