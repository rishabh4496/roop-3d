import { lazy, Suspense } from 'react';
import { MotionDrawer } from '../components/ui-layouts';

// The two deep-linked drawers (#/models, #/diagnostics). Their bodies load on first open, so the Studio's first paint is unchanged.
const ModelManager = lazy(() => import('../components/modelhub/ModelManager'));
const ModelHubPanel = lazy(() => import('../components/modelhub/ModelHubPanel'));
const RuntimeReportPanel = lazy(() => import('../components/diagnostics/RuntimeReportPanel'));

const Loading = ({ what }: { what: string }) => <p className="muted empty-note" role="status">Loading {what}…</p>;

/** Closing a drawer returns to the Studio, so the address bar never keeps pointing at something that is no longer open. */
const closeDrawer = () => { if (window.location.hash !== '#/studio') window.location.hash = '#/studio'; };

export default function ShellDrawers({ drawer }: { drawer: string | null }) {
  return (
    <>
      <MotionDrawer open={drawer === 'models'} onClose={closeDrawer} title="Model Hub" description="Local models: what is installed, verified and selected for each stage." size="min(620px, 96vw)">
        <Suspense fallback={<Loading what="the model hub" />}><ModelManager /><ModelHubPanel active /></Suspense>
      </MotionDrawer>
      <MotionDrawer open={drawer === 'diagnostics'} onClose={closeDrawer} title="Runtime report" description="Pinokio runtime report: hardware, execution providers, precision and model state, section by section." size="min(560px, 96vw)">
        <Suspense fallback={<Loading what="the runtime report" />}><RuntimeReportPanel /></Suspense>
      </MotionDrawer>
    </>
  );
}
