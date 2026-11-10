# Articulated reconstruction models

The source files `lbs.py`, `monocular_regressors.py`, and their unit tests are
original implementations licensed under `ARTICULATED_MODELS_LICENSE` (MIT).
They contain no imported research-project implementation, model checkpoint,
training data, or face geometry asset.

Install the optional model dependency from `app/roop_3d` with
`pip install -e ".[articulated-reconstruction]"`. PyTorch is deliberately
optional for the rest of the Roop 3D application.

## Interfaces

`ArticulatedHeadLayer` takes metric template geometry, five skinning-weight
columns (root, neck, jaw, left eye, right eye), and optional shape, expression,
and pose-corrective bases. The articulated groups are root, neck, jaw, and a
shared six-DoF eye pose. Inputs are source `beta (B, 300)`, target `psi (B, 100)`,
three axis-angle vectors for root/neck/jaw, and `theta_eyes (B, 6)` containing
eye axis-angle followed by eye translation in metres. Source identity is locked
at 1.0 by default. An optional `beta_target` is used to convert target expression
displacements into source metric scale; it does not alter source geometry while
the lock remains 1.0. The layer returns metric vertices and yaw-selected
cheek-contour points.

`ArcFaceToMetricShapeAdapter.forward_faceset` accepts a parsed `.fsz`
`FaceSetDocument` or a sequence of its views. It reads each stored 512-D
ArcFace embedding, sharpness score, and yaw. The tensor-only `forward` method
is the export interface and returns 300 shape coefficients.

`AsymmetricExpressionEncoder` predicts 100 expression coefficients and direct
per-vertex residuals. `LipPhonemeRegressor` predicts a frame-local jaw pose,
inner-lip residuals, and silence/P/B/M/A/E/I/O/U logits. Neither branch has a
temporal filter. `MonocularArticulatedHead` composes these regressors, the
identity adapter, and LBS into one ONNX graph. Its forward and exported graph
accept target beta explicitly so expression scale is always normalized.

## Training and export

All learned output layers start at zero so an untrained instance returns a
neutral pose and zero shape/expression offsets. This is an architecture, not a
pretrained estimator: meaningful identity, expression, pose, and phoneme
predictions require trained weights and suitable licensed supervision.

Export the composed graph with `export_articulated_head_onnx(model, path)`.
Batch and reference-view counts are dynamic; face and mouth crop resolutions
are fixed in the exported optimization profile. The ONNX graph is suitable as
input to the optional TensorRT runtime. Model assets and the TensorRT engine
are expected to be managed by the application and are not shipped here.
