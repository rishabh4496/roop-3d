// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import * as viewport from './ThreeViewport';

// StudioInspector and App wrap these exports in `lazy(() => import('./ThreeViewport').then(m => ({ default: m.X })))`.
// A `lazy` result is an object, so re-exporting one here makes React throw "Lazy element type must resolve to a
// class or function" and the Inspector panel fails to render.
describe('ThreeViewport module exports', () => {
  it('exports plain components that can be wrapped in lazy() by consumers', () => {
    const exported: Record<string, unknown> = {
      PoseGeometryViewer: viewport.PoseGeometryViewer,
      LightProbeSphere: viewport.LightProbeSphere,
      FlameMeshViewer: viewport.FlameMeshViewer,
      SpatialInspector: viewport.SpatialInspector,
      ThreeViewport: viewport.ThreeViewport,
      PoseCoverageInspector: viewport.PoseCoverageInspector,
    };
    for (const [name, component] of Object.entries(exported)) {
      expect(component, name).toBeTruthy();
      const lazyObject = typeof component === 'object' && component !== null && '$$typeof' in component
        && component.$$typeof === Symbol.for('react.lazy');
      expect(lazyObject, `${name} must not be a React.lazy object`).toBe(false);
    }
  });
});
