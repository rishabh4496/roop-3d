import type { ReactNode } from 'react';
import type { Job } from '../contracts';
import type { WorkflowProgress } from '../components/guide/workflowState';
import type { WorkflowStepId } from '../components/guide/helpContent';
import WorkflowChecklist from '../components/guide/WorkflowChecklist';
import { Reveal, Stagger, TiltCard } from '../motion';
import { IconActivity, IconFolder, IconSwap, IconUsers } from '../shell/icons';

export interface HomeProps {
  sourceCount: number; jobs: Job[]; projectCount: number;
  workflow?: WorkflowProgress;
  onWorkflowAction?: (id: WorkflowStepId) => void;
  onStartTour?: () => void; onOpenGuide?: () => void; onOpenLibrary: () => void;
  recentProjects: ReactNode;
}

const baseName = (path: string | undefined) => (path ?? '').split(/[\\/]/).pop() || 'output';

function Stat({ href, icon, value, label, hint }: { href: string; icon: ReactNode; value: number | string; label: string; hint: string }) {
  return (
    <Reveal>
      <TiltCard className="stat-card">
        <a href={href} className="stat-link">
          <span className="stat-icon">{icon}</span>
          <span className="stat-value">{value}</span>
          <span className="stat-label">{label}</span>
          <small>{hint}</small>
        </a>
      </TiltCard>
    </Reveal>
  );
}

/** Launch pad: where to start, what is running, what you made. Everything on this screen links somewhere useful. */
export default function HomeScreen(props: HomeProps) {
  const active = props.jobs.filter(job => ['queued', 'preparing', 'running'].includes(job.state)).length;
  const done = props.jobs.filter(job => job.state === 'succeeded' && job.output);
  const latest = [...done].reverse().slice(0, 4);
  return (
    <main className="screen-page home-screen">
      <Stagger className="home-stack">
        <Reveal>
          <section className="home-hero">
            <div>
              <span className="eyebrow">LOCAL CREATIVE WORKSPACE</span>
              <h1>Dashboard</h1>
              <p className="muted">Swap faces in your own videos. Everything runs on this computer: your media and results are never uploaded.</p>
            </div>
            <div className="home-help-actions">
              <a className="button accent hero-cta" href="#/faceswap"><IconSwap size={16} /> Start a scene</a>
              <button className="button tour-start" data-testid="home-tour-button" onClick={() => props.onStartTour?.()}>How to Use · Quick Tour</button>
              <button className="button" data-testid="home-guide-button" onClick={() => props.onOpenGuide?.()}>Open the guide</button>
            </div>
          </section>
        </Reveal>

        <div className="stat-grid">
          <Stat href="#/facemgr" icon={<IconUsers />} value={props.sourceCount} label="Source identities" hint="Build and inspect FaceSets" />
          <Stat href="#/processing" icon={<IconActivity />} value={active} label="Active jobs" hint="Live local queue" />
          <Stat href="#/gallery" icon={<IconFolder />} value={done.length} label="Finished renders" hint="Browse your outputs" />
          <Stat href="#/history" icon={<IconSwap />} value={props.projectCount} label="Saved projects" hint="Pick up where you left off" />
        </div>

        <div className="home-columns">
          {props.workflow && (
            <Reveal className="home-col">
              <section className="home-workflow panel-card" aria-labelledby="home-workflow-title">
                <div className="panel-card-head"><span className="eyebrow">GETTING STARTED</span><h2 id="home-workflow-title">Your video workflow</h2></div>
                <WorkflowChecklist progress={props.workflow} onAction={props.onWorkflowAction} compact />
              </section>
            </Reveal>
          )}
          <Reveal className="home-col">
            <section className="panel-card" aria-labelledby="home-recent-title">
              <div className="panel-card-head row"><div><span className="eyebrow">RESUME</span><h2 id="home-recent-title">Recent projects</h2></div><button className="button small" onClick={props.onOpenLibrary}>Project library</button></div>
              {props.recentProjects}
            </section>
            <section className="panel-card" aria-labelledby="home-outputs-title">
              <div className="panel-card-head row"><div><span className="eyebrow">RESULTS</span><h2 id="home-outputs-title">Latest outputs</h2></div><a className="button small" href="#/gallery">All outputs</a></div>
              {latest.length ? <ul className="output-list">{latest.map(job => (
                <li key={job.job_id}><IconFolder size={16} /><strong title={job.output?.output_path}>{baseName(job.output?.output_path)}</strong></li>
              ))}</ul> : <p className="muted empty-note">Finished renders appear here.</p>}
            </section>
          </Reveal>
        </div>
      </Stagger>
    </main>
  );
}
