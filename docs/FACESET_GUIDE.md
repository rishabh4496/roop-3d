# FaceSet creation guide

A **FaceSet** is the identity you put into a video: several photos of *one* person from different head angles, stored with measured face data. A
good FaceSet matters more than any setting: the swap can only reproduce angles it has seen.

This guide mirrors the checks Roop 3D really applies (`reconstruction/faceset_quality.py` and `faceset/faceset_reader.py`), so the numbers below
are the actual limits, not rules of thumb.

## 1. Ways to get a FaceSet

| Way | Where | Good for |
|---|---|---|
| Import an archive | Face Swap → *02 · Identity* → **Import .fsz** (or Face Manager → **Import Source Photos** / FaceSet Library) | A FaceSet you or someone you trust already made |
| Build from photos | **Face Manager → 3D Studio** → **Import Source Photos**, review the coverage, **Save to FaceSet Library ↗** | The normal route |
| Take a face from your video | Scrub to a clear frame, right-click the face, **Use Face From Frame** (or press **E**) | When you only have the video |
| Fill gaps from the video | Face Manager → select the identity → **Fill Missing Angles from Current Target Video** | Adding the angles your photos lack. Only observed faces that match the selected identity are added; long scans stop at the reported frame or time budget |
| Export | **Download .Fsz** | Backing up or moving a FaceSet |

## 2. Choose the photos

Collect **5 to 12 photos of one person**. More photos only help when they add a new angle or expression.

### Photo checklist (each is a real rule)

| Rule | The actual limit | Why it matters |
|---|---|---|
| Sharp and in focus | A photo is rejected as `TOO_BLURRY` when its sharpness is below **0.03** (on a 0–1 scale that is the variance of fine detail divided by 500) | Blurry photos give the swap nothing to copy. Motion blur and heavy compression both count |
| Face at least 40 pixels wide | Smaller faces are rejected (`FACE_TOO_SMALL`) | Aim for 256 px or more across the face |
| Whole face inside the picture | More than 35% of the face box outside the image → `FACE_TRUNCATED_BY_IMAGE_EDGE` | Cropped faces lose the jaw or forehead |
| Even exposure | Mostly black or burnt-out photos are rejected (`BAD_EXPOSURE`) | Avoid hard shadows across one side of the face |
| Nothing in front of the face | Heavy occlusion is excluded (`FACE_HEAVILY_OCCLUDED`) | Hands, hair, masks, sunglasses hide what the swap needs. Glasses are fine only if the target person also wears them |
| Only one person | A photo whose face is too different from the rest (similarity below **0.30**) is dropped as an outlier (`IDENTITY_OUTLIER`); if the set splits between people the whole build is blocked (`blocked_inconsistent_identity`) | One wrong person spoils the identity |
| No near-duplicates | Two photos that look the same *and* share a head angle (within 3°) count once (`DUPLICATE_IMAGE`) | Duplicates add weight, not information |
| A detectable face | If no face is detected the photo is not used as an observed view; landmarks are never invented for a 3D build (`LANDMARKS_NOT_OBSERVED`) | The 3D build only uses measured faces |
| Head turned no more than 88° | Beyond that: `POSE_OUT_OF_RANGE` | Almost-sideways and backward views cannot be measured |

A photo that fails is kept in the FaceSet but not used for the 3D build.

## 3. Cover the angles

Yaw is the left-right turn of the head, pitch the up-down tilt. "Left" and "right" are as seen **in the picture** (the direction the nose points).

| Angle | Target | Importance |
|---|---|---|
| Frontal | 0° | **Required** |
| Three-quarter left | −45° | **Required** |
| Three-quarter right | +45° | **Required** |
| Left profile | −75° | Recommended |
| Right profile | +75° | Recommended |
| Looking down / up | −20° / +20° pitch | Optional, for vertical coverage |
| Expression | A neutral/closed-mouth and an open-mouth view | Optional. If every view shares one expression, Roop 3D asks for both so identity and expression can be separated |

A view counts for an angle when it is within **20°** of the yaw target (**12°** for pitch).

### Reading the coverage grid

The grid in the Identity panel and in Face Manager has five yaw columns (−75°, −45°, 0°, +45°, +75°) and three pitch rows (Up, Level, Down).
Each cell shows how many views fall in it: **–** none, **1** sparse, **2+** well covered. A sentence under the grid says what is still needed
(for example *"1 of 15 angles covered. Still needed: three-quarter left (−45°) and three-quarter right (+45°)."*). Cells are never shown by colour alone.

The overall **coverage label** is one of:

| Label | Meaning | Next step |
|---|---|---|
| Not enough usable photos | Too few views passed the quality checks | Add sharp, front-facing photos |
| Blocked: more than one person | The photos do not agree on who this is | Remove photos of other people, then rebuild |
| Front only | Only straight-on views exist: swaps work while the person faces the camera and degrade when they turn | Add three-quarter left and right |
| Partial coverage | Some turned angles are covered | Add the angles under *suggested additional angles* |
| Full coverage | Frontal and both three-quarter views are present | Nothing required (profiles are optional) |

*Suggested additional angles* lists concrete requests in priority order, tagged **required**, **recommended** or **optional**.

## 4. The 3D identity asset

From the FaceSet detail, **Build 3D identity asset** reconstructs a 3D face from the FaceSet's own views, on the CPU, usually in seconds. It is
also built automatically the first time a 3D render mode needs it. **View 3D asset** shows the mesh in the Inspector. The panel also lists *what
this asset does not contain* (for example the back of the head) so you know its limits. The two 3D render modes use this asset; the standard
render mode does not.

## 5. Face Manager's 3×3 pose matrix

Face Manager arranges a person's photos in a 3×3 matrix (yaw −1 / 0 / +1 by pitch −1 / 0 / +1). A cell reads **Missing** until a photo covers it,
and shows the best photo's measured sharpness and quality when one does. *Local Photo Cosine Distance* (default 0.38) sets how different two
photos' faces may be and still be grouped as the same person; lower it if different people are merged, raise it if one person is split.

## 6. Quality score

When several photos fit one angle, Roop 3D prefers the best by a **quality score** = 0.7 × sharpness + 0.3 × detection confidence; a view needs a
score of at least 0.5 (and a detection confidence of at least 0.5) to count towards the identity's canonical embedding.

## 7. Problems

| You see | Do this |
|---|---|
| `FACESET_EMPTY` | No usable face in the images. Use sharp, front-facing photos |
| `FACESET_ARCHIVE_INVALID` / `…_SECURITY_VIOLATION` | The `.fsz` is damaged or unsafe. Rebuild from photos |
| Coverage says *blocked: more than one person* | Remove the other person's photos |
| The swap looks wrong when the person turns | The FaceSet lacks that angle: check the grid and add it |
| Eyes or teeth look wrong | Add a neutral and an open-mouth photo; try the restoration model ([Engine comparison](ENGINE_COMPARISON.md)) |

See also: [User manual](USER_MANUAL.md) · [Common problems](COMMON_PROBLEMS.md#facesets)
