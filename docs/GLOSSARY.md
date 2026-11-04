# Roop 3D Orchestration

Language for choosing and coordinating the AI components that make up a face-swap processing run.

## Language

**Model**:
A governed, hash-verified neural-network artifact with recorded provenance, license, and hardware/precision requirements.
_Avoid_: Weights, checkpoint, network

**Engine**:
One orchestrator-selectable provider of a single capability category, composed of one or more Models, an adapter, and a declared input/output contract.
_Avoid_: Backend, provider, plugin

**Pipeline plan**:
The set of Engines chosen to process one scene segment of one track.
_Avoid_: Workflow, configuration

**Render profile**:
A named set of allowed Engines per category, required or optional, from which Pipeline plans are formed. The baseline profile allows exactly one Engine per category.
_Avoid_: Mode (reserved for the execution mode)

**Job runner**:
The existing whole-job executor that a job is dispatched to by identifier.
_Avoid_: Engine (reserved for the capability provider)

**Shot**:
One continuous stretch of the target between hard cuts. All tracks in a shot share one Pipeline plan.
_Avoid_: Segment, scene (when meaning the plan unit)

**Evidence record**:
A measurement of one concrete Pipeline plan on one hardware tier, made with real Models. Only an exact match counts for a plan.
_Avoid_: Benchmark, score

**Baseline plan**:
The pinned plan equal to today's standard reference swap; it is the fallback and the quality floor.
_Avoid_: Default plan

**Unavailable Engine**:
A well-formed Engine that cannot be used on a tier because its adapter, governance state or tier support is missing.
_Avoid_: Disabled engine

# Roop 3D Media and Colour

Language for how media is read, interpreted and written without changing what it looks or sounds like.

**Colour description**:
The resolved primaries, transfer, YCbCr matrix, range, bit depth and chroma layout of a video stream. Each field is either tagged or assumed.
_Avoid_: Colour space (ambiguous: it can mean gamut, matrix or both)

**Tagged / assumed**:
A field the file states, versus one resolved by a named convention and recorded as an assumption. Untagged material is never assumed to be HDR.
_Avoid_: Default, guessed

**Working colour**:
What the 8-bit RGB frames handed to a transform mean, and how an output must be tagged. For SDR it equals the source; for HDR it is BT.709 SDR plus the name of the operator that produced it.
_Avoid_: Pipeline colour space

**HDR policy**:
What PQ or HLG input becomes: tone-mapped to SDR, refused, or (only when no pixel is processed) passed through.
_Avoid_: HDR mode

**Passthrough**:
Re-encoding the source's own YCbCr planes without an RGB round trip. The only way PQ/HLG material leaves the system as HDR.
_Avoid_: Copy, remux (it re-encodes)

**Native path**:
The encode path that takes decoded planes directly. The alternative is the RGB path, used whenever a transform changes pixels.
_Avoid_: Fast path

**Export profile**:
A named, tested combination of container, codec, pixel format and bit depth that states what it can carry and never promises more.
_Avoid_: Preset, quality setting

**Processing bit depth**:
The precision at which faces are detected, swapped and composited (8 bits). Distinct from the container bit depth, which a 10-bit profile raises without adding precision.
_Avoid_: Bit depth (unqualified)

**Linear-light blend**:
Alpha arithmetic performed on linear light, using a named display transfer, instead of on encoded code values.
_Avoid_: Gamma-correct blending

**Fidelity check**:
Comparison of frames sampled during encode with the same frames decoded from the finished file, to catch range, matrix or depth errors that frame count and PTS cannot.
_Avoid_: Quality check

**Signal lag**:
The offset between source and output audio measured by decoding both onto one timeline and correlating them. Zero for a stream copy.
_Avoid_: Sync offset (container timestamps can agree while the audio does not)
