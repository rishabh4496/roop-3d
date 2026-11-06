import { createRoot } from 'react-dom/client';
import FacesetBuilder from '../../src/components/facemgr/FacesetBuilder';
import '../../src/styles.css';
createRoot(document.getElementById('root')!).render(<FacesetBuilder sourceId="existing" onSaved={id => { document.getElementById('saved')!.textContent = id; }}/>);
