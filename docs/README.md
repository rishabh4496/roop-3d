# Roop 3D documentation

## For people using Roop 3D

| Read this | When |
|---|---|
| [Installation guide](INSTALLATION_GUIDE.md) | Installing, starting, updating or resetting the app |
| [Offline model installation](OFFLINE_MODELS.md) | Putting the model files on a computer without internet; what the safety checks mean |
| [User manual](USER_MANUAL.md) | Every screen and control, the workflow, shortcuts, accessibility |
| [FaceSet creation guide](FACESET_GUIDE.md) | Choosing photos, covering head angles, reading the coverage grid |
| [Engine comparison guide](ENGINE_COMPARISON.md) | Engines, render modes, restoration, hardware and provider choices |
| [GPU troubleshooting guide](GPU_TROUBLESHOOTING.md) | The GPU is not used, out-of-memory errors, hardware profiles |
| [Performance guide](PERFORMANCE_GUIDE.md) | Making renders faster; what costs time; how to measure |
| [Export guide](EXPORT_GUIDE.md) | Render presets, HDR and subtitles, passes, finishing, provenance |
| [Common problems and solutions](COMMON_PROBLEMS.md) | Something went wrong: by symptom, and by error code |

Inside the app, press **Guide** in the header for the same material, tied to what you are doing.

## For people changing Roop 3D

| Read this | When |
|---|---|
| [Developer architecture](ARCHITECTURE.md) | Layout, processes, API, jobs, the interface, tests, how to add guidance |
| [Terminal HUD, runtime report and TensorRT heartbeat](TERMINAL_HUD_AND_RUNTIME_REPORT.md) | What the Pinokio terminal prints, the 14-section report, why a cold TensorRT build freezes the server, and how the heartbeat works around it |
| [Stage 23: workstation parity](STAGE23_WORKSTATION_PARITY.md) | Multi-person actors, paint/erase frame ranges, GOP batch planning, the 10 job lifecycle states: what existed, what was added, what is still not produced |
| [Stage 19 report](UX_STAGE19.md) | What the usability and accessibility review found, fixed, and measured |
| [Hardware and dependency matrix](HARDWARE_COMPATIBILITY_MATRIX.md) | Verified versions and hardware profiles |
| [Reliability (Stage 18)](RELIABILITY_STAGE18.md) | Checkpoints, recovery, soak evidence and limits |
| [Post-production](POST_PRODUCTION.md) | Passes, interchange, finishing, provenance: engineering detail |
| Earlier stage documents | `STAGE*`, `REMEDIATION_STAGE*`, `*_STAGE*.md` in this folder |

The user documents are checked against the code by `app/roop_3d/tests/test_stage19_docs.py`: a documented number, code, command, preset, default
or on-screen label that stops matching the application fails that test.
