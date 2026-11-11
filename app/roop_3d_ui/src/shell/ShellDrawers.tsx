import { MotionDrawer } from '../components/ui-layouts';
import ModelManager from '../components/modelhub/ModelManager';
import ModelHubPanel from '../components/modelhub/ModelHubPanel';
import RuntimeReportPanel from '../components/diagnostics/RuntimeReportPanel';

/** Closing a drawer returns to the Studio, so the address bar never keeps pointing at something that is no longer open. */
const closeDrawer = () => { if (window.location.hash !== '#/studio') window.location.hash = '#/studio'; };

export default function ShellDrawers({ drawer }: { drawer: string | null }) {
  return (
    <>
      <MotionDrawer open={drawer === 'models'} onClose={closeDrawer} title="Model Hub" description="Local models: what is installed, verified and selected for each stage." size="min(620px, 96vw)">
        <ModelManager /><ModelHubPanel active />
      </MotionDrawer>
      <MotionDrawer open={drawer === 'diagnostics'} onClose={closeDrawer} title="Runtime report" description="Pinokio runtime report: hardware, execution providers, precision and model state, section by section." size="min(560px, 96vw)">
        <RuntimeReportPanel />
      </MotionDrawer>
    </>
  );
}
