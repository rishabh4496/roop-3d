# Stage 24: liquid-glass HUDs, drag-reorder lists, Embla frame dock

The prompt's `app/roop_3d_ui/src/` is real. The upstream source for the `ui-layouts` components is not vendored here and this app
has no Tailwind, so each component is a re-implementation of the documented props (read from `uilayouts/apps/ui-layout/content/
components/*.mdx`) on this app's CSS tokens, using the pinned `framer-motion` and one new pinned dependency, `embla-carousel-react@8.6.0`.

## Components (`src/components/ui-layouts/`)

| Component | What it does | Where it is used |
|---|---|---|
| `LiquidGlassCard` | `draggable`, `expandable`, `blurIntensity` sm..xl, `glowIntensity`, `shadowIntensity`, elastic spring back. Extra: `dragHandle` (only a grip drags) | Viewport telemetry pill, before/after loupe HUD, brush toolbar |
| `DragItems` | Spring reorder on `Reorder`; `onReorder` follows the drag, `onCommit` fires once on drop; Alt+arrow keys and move buttons; polite announcement | Persistent queue (`#/processing`), recipe order (`#/batch`) |
| `EmblaCarousel` | Draggable inertial strip with prev/next buttons and "follow this slide". Falls back to a plain scrolling row where `window.matchMedia` is missing (Embla throws there) | Frame dock |
| `TimelineAnimation` | Staggered scroll reveal (`animationNum`, `delay`, `customVariants`, `viewport`). Shows content at once under reduced motion or without `IntersectionObserver` | Dock ruler, queue lifecycle strip |

Deliberate deviation: upstream's default reveal margin (`-120px` at the bottom) would keep anything docked near the window's bottom edge
invisible forever, so the default is `{ amount: 0.05, margin: '0px' }`.

## Screens

* **Floating HUDs on the full-bleed canvas** (`CanvasHuds.tsx`): the telemetry pill (yaw/pitch/roll of the selected face, face count, frame,
  mode), a before/after lens pair (two 4x lenses, "Before/After lens" toggle) and the brush toolbar on a draggable glass card.
  Dragging the toolbar by its grip never paints; releasing springs it back.
* **`#/processing`**: `BackendQueuePanel` shows the persistent queue with each job's `lifecycle_state` chip and strip, drag-to-reorder for
  `queued`/`paused` jobs only (a drop never relocates a running job; `reorderPayload` keeps their slots), and pause/resume/cancel/
  duplicate/join/retry-segment. It reads `/api/v1/queue`, a different store from the older list above it.
* **`#/batch`**: drag the recipe chips to reorder the matrix columns; jobs are queued in chip order.
* **Frame dock (`TimelineDock`)** at the bottom of the Studio and the Editor: Embla filmstrip with a measured ruler label per frame, In/Out
  trim handles (pointer or keyboard: arrow = 1 frame, Shift = 1 s, Home/End), chapter markers (same store as the timeline, refreshes via
  `roop3d:chapters`), and 0.25x-4x speed. Speed is one shared store (`playbackRateStore`): the long timeline's own buttons, the dock and
  the finished-video preview in the Editor all follow it. The long timeline stays; the dock is a compact second view of the same
  playhead and trim, not a replacement.

## Bugs found while verifying (all fixed)

* `timelineDock.ts` vs `TimelineDock.tsx` collide on a case-insensitive filesystem (renamed `dockMath.ts`).
* The dock's filmstrip overflowed its panel (grid children need `min-width: 0`); found by looking at a screenshot.
* The brush toolbar overflowed its glass card and collided with "Done Mask"; it now wraps.
* `Embla` and framer's reveal crash without `matchMedia` / `IntersectionObserver` (see the fallbacks above).
* `queue_manager.reorder` accepted unknown/duplicate ids and `duplicate` copied segment states (Stage 23 file, fixed there).

## Verification

See the commit message for the gate results. Browser checks: `e2e/stage24.e2e.ts` (8 tests, real Chrome, mocked service).
Screenshots: `test-results/stage24/`. Known limits: thumbnails in the browser tests are the mock's placeholder images, and the 3D mesh
canvas mode is the existing orientation overlay.
