# Stage 19: professional UX, accessibility and documentation

Goal: a person who processes their own offline video can complete the whole local workflow without understanding the Python architecture.
Constraint: no redesign for novelty; fix what is actually hard, broken or inaccessible, and prove it in a real browser.

## 1. Method

1. Built the interface and ran the **real server** (`serve.py`) and **real browser** (system Chrome, Playwright) through every screen at five viewports
   (1680×1050, 1366×768, 1024×768, 820×1180, 768×1024), with real data (real FaceSets, a real video).
2. Measured instead of guessing: layout overflow, unnamed controls, WCAG 2.2 AA with **axe-core 4.10.3**, keyboard focus, theme contrast in all 37 themes.
3. Fixed the causes, wrote tests that fail without the fix (`e2e/stage19.e2e.ts`, `src/stage19.test.ts`, `tests/test_stage19_*.py`).
4. Wrote the documentation from the code and added `test_stage19_docs.py` so it cannot drift.

## 2. What was added (features)

| Requirement | Delivered | Where |
|---|---|---|
| First-run onboarding | A *Welcome* dialog on the first launch (what it does, "never uploaded", live model/hardware status, four ways forward), shown once | `WelcomeDialog` |
| Guided project creation | **Start a guided project** → workstation with the project name selected; a **Next-step bar** (one sentence, one button) and a live **checklist** (Home and Guide) driven by real state | `NextStepBar`, `WorkflowChecklist`, `workflowState.ts` |
| Target-face selection tutorial | In-place *"Selecting target faces: how does this work?"* + empty-state guidance + Guide tab | `TargetPanel`, `GuideDialog` |
| FaceSet preparation guide, source image quality guidance | Source photo checklist with the **real thresholds** (sharpness 0.03, 40 px, 35% edge, 0.30 identity, 88°), Guide tab, `docs/FACESET_GUIDE.md` | `FaceSetDetail`, `helpContent.ts` |
| 3D coverage visualisation | Existing grid made accessible (never colour-only: counts, labels), plus a plain-language sentence ("still needed: …"), a legend, the readiness meaning and **prioritised** requests (required / recommended / optional) | `coverageGuide.ts`, `FaceSetDetail` |
| Recommended engine explanation | A **Recommended** line with the reason, the chosen engine's status in words, and why unavailable engines cannot be chosen | `EngineExplainer` |
| Hardware profile explanation | What the resolved profile means and how to read the CPU/TRT/VRAM pill, in the Inspector and Settings → Compute | `HardwareExplainer` |
| Render preset explanation | **New control** (it did not exist, see §3) with every preset explained: use, size, container, bit depth, limits; HDR, subtitle and image-format policies | `ExportPresetBar` |
| Queue and checkpoint guidance | In the drawer; per-state hints; an honest note that resume replays the analysis from frame 0 | `QueueDrawer`, `helpContent.ts` |
| Clear recovery instructions | Backend **error codes are kept** (`ApiError`) and mapped to *what happened / try this first / every step*; request IDs shown; 49 codes documented | `errorGuide.ts`, `RecoveryNotice`, `docs/COMMON_PROBLEMS.md` |

### Accessibility

| Requirement | Delivered |
|---|---|
| Keyboard navigation | Skip link; every control reachable; tabs with arrow/Home/End; one shared dialog contract (`useDialogA11y`) |
| Focus visibility | A 3 px accent ring + halo on everything focusable, including scroll containers; verified for 60 consecutive Tab stops |
| Screen-reader labels | Every interactive element named on all nine screens (test); `role`/`aria` repaired (see §3); progress bars expose values; page title per screen |
| Appropriate contrast | **Contrast enforcement** (`themeContrast.ts`) keeps muted/accent/danger/button text ≥ 4.5:1 in all 37 themes (including user-made ones); component fixes for light and AMOLED themes; axe finds **0 violations** on the workstation in Default, Glass Light, AMOLED Black and Arctic Light |
| Scalable UI | Existing zoom (50–160%) kept and surfaced; reflow verified down to 640 px (200% zoom) |
| Clear error messages | See recovery above |
| Consistent controls | Minimum 24×24 px targets (WCAG 2.5.8, axe `target-size`); shortcut labels now match the real pipeline nodes (see §3) |
| Accessible dialogs | Six dialogs previously had no/partial semantics or focus handling; all now labelled, modal, trapped, Esc-closable, focus-restoring |
| Motion-reduction option | **Reduce motion**: follow system / always / never. Stops CSS animation *and* the 3D idle rotation, camera easing and smooth scrolling |

### Responsive behaviour

Desktop and tablet were tested at the five viewports above with screenshots of every screen (`test-results/stage19/`). At ≤1279 px the **frame comes first** and
the chrome above it is slimmed: at 1024×768 the preview now starts at y≈280 and is fully on screen (it started at y≈500, mostly below the fold).
**Preview resolution is untouched**: a test asserts that the preview requests are identical at 1680, 1024 and 820 px wide.

## 3. Defects found and fixed

Found by running the product, not by reading it. Each has a regression test.

| # | Defect | Fix |
|---|---|---|
| 1 | **A real "12 GB" GPU was silently demoted to the 6 GB profile.** An RTX 4070 reports 12282 MiB; the profile option required 12288 and render admission downgraded; *Auto* and the preflight said `12gb` | One threshold, the documented 10,240 MiB (`MIN_12GB_PROFILE_VRAM_MIB`) in all three places. `test_stage19_hardware_profile.py` |
| 2 | **Offline model import put files where the server never looks.** `project_admin.py model-import/-verify/offline-health` used the project root, the server uses `app/` | They now use `APP_ROOT`. `test_stage19_cli_model_root.py` |
| 3 | **The whole app was unusable for ~11 s after load.** `connect()` waited for `/projects`, which re-hashes every project's inputs (10.8 s here) | `/projects` loads separately; "Checking saved projects…" instead of a false "No saved projects" |
| 4 | A silent no-op: pressing Import before capabilities arrived did nothing | A message, and the next-step button is `aria-disabled` with the reason |
| 5 | The **Face Manager overflowed sideways** at tablet widths (990 px page in an 820 px window) | Stacked at ≤1100 px |
| 6 | The preview toolbar was clipped on its left edge on narrow panes | Wraps inside the pane |
| 7 | The project name field was clipped ("Untitled pr") | Sized properly; header fits one row on desktop |
| 8 | A malformed Pre-Flight Linter answer **blanked the entire page** | Validated: a visible, retryable error |
| 9 | The Pre-Flight Linter dialog had no dialog role, label, focus management or Esc | Shared dialog contract |
| 10 | The Editor's "Swapped-Face-Only Foreground (RGBA)" pass could be ticked but the server rejects it, so the render failed validation | The checkbox is disabled with the reason |
| 11 | Shortcut labels were wrong ("3 = Raw Swap" soloed *3D Normal Map*); stages 7–8 had no key | Labels match the pipeline; keys 7 and 8 added |
| 12 | Accent buttons turned unreadable on hover (dark ink on dark green) | Hover keeps contrast |
| 13 | `.mint`/`.amber` background classes were used on text: 1.3:1 contrast | Treated as text colour on non-dots |
| 14 | `var(--panel,#151c20)` / `var(--border,…)` referenced undefined tokens, so the Editor stayed dark with dark text in every light theme | Tokens defined |
| 15 | Light themes printed dark text on dark timeline strips (1.2:1); hard-coded neon/amber on light surfaces | Local dark palette for the strips; darker text on light surfaces |
| 16 | 60+ controls below 24×24 px (sliders 18 px high, 8–9 px speed/mode buttons, 12 px marker pins) | Minimum sizes; marker hit area enlarged without changing the dot |
| 17 | Invalid ARIA: `aria-rowcount` on a list, `grid` without rows, `tablist` without tabs, a labelled `div`, an empty `list` | Repaired |
| 18 | "Reconnect" was offered for every error | Only for connectivity errors |
| 19 | Preset wording leaked internals ("pre-Stage-15 default") | Plain wording |

## 4. Evidence

| Check | Result |
|---|---|
| `npm run build` (typecheck + lint + build) | 0 errors |
| Vitest | 42 files, 369 tests pass (28 new in `stage19.test.ts`) |
| Playwright, Stage 19 spec (`e2e/stage19.e2e.ts`) | 34 tests pass: onboarding, guided flow, guidance, presets round-trip, errors, 11 dialogs/keyboard/focus/motion/zoom checks, axe on 4 screens + 7 guide tabs + welcome + 3 themes, contrast in all themes, 5 viewports × 9 screens, preview-resolution invariance |
| Playwright, whole suite | 60 of 60 pass, including the earlier specs (identity lock, studio, telemetry, WebGL, resource leaks, swap audit) after updating one expectation for the changed copy and renaming the Next-step render button so it does not duplicate "Queue render" |
| Python | 28 Stage 19 tests (`test_stage19_hardware_profile.py`, `test_stage19_cli_model_root.py`, `test_stage19_docs.py`) plus 143 related existing tests; the whole Python suite was not re-run (it is long and must run in chunks). One Stage 18 recovery test failed once in a combined run and passed alone and on rerun |
| Real-server run | Real FaceSets, preset list and hardware profile were read live; a real 7-minute video imported and a 12 s clip loaded through the guided flow |

## 5. Known limits (stated plainly)

1. **Face inspection cannot complete on this build.** The model registry shipped with this checkout records licence, security, precision and provenance
   as `UNKNOWN` for the core models, so the governance gate refuses to run them (`LICENSE_UNVERIFIED`), even with every file present and verified.
   No interface control can clear it. This is a deliberate rights control from an earlier stage and was **not** changed. Consequently the
   *end-to-end render on a real video could not be demonstrated here*: the real-server run reached "Inspect target faces", where the new
   guidance now explains the refusal. Everything after inspection was verified against the mocked service in a real browser. The documentation states the limit
   ([Offline models §4](OFFLINE_MODELS.md#4-model-review-why-a-verified-file-can-still-be-refused)).
2. `GET /projects` still takes seconds with many large saved projects; the interface no longer waits for it, but the server re-verifying every project on each call
   is a candidate for caching.
3. CodeFormer's fidelity control is marked inactive by the app itself ("not wired into the preview compositor yet"); the docs say so.
4. Resume replays the analysis from frame 0 (documented, from Stage 18).
5. The 30/60-minute soak runs from Stage 18 remain open; nothing here changes that.
6. Screen readers were not run (no NVDA/JAWS/VoiceOver here); semantics were verified with axe, accessible-name audits and keyboard tests, which cannot prove the spoken experience.
7. WCAG conformance is claimed only for what was measured (axe + the checks above); axe finds roughly a third of possible issues.
