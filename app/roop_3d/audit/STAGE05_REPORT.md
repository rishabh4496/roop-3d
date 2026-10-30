# Stage 05: video/audio I/O, rational timestamps and keyframes

Implemented only in the standalone application's `app/roop_3d/` package. The
stage paths are relative to its existing `app/` application root. No launchers,
root policy files, dependency locks, sibling source, or preserved baselines changed.
No commit or push was requested or performed.

## Delivered behavior

- `media/probe.py`: local-only, link-rejecting PyAV probe. Records container,
  all video/audio stream indices/codecs, Fraction time bases, declared frame
  counts, rational nominal/average rates, duration/start ticks, pixel format,
  color identifiers (including BT.709/BT.601 labels), display rotation, and
  audio layout/rate/language. Timing sampling has frame/time budgets and distinguishes
  observed VFR, completely observed CFR, and UNKNOWN for partial constant samples.
- `media/video_reader.py`: deterministic pull-based BGR/RGB decoding, bounded
  frame allocation, bounded pending packet metadata, keyframe-only iteration,
  exact keyframe seeking followed by frame selection, and bounded JPEG thumbnails.
  CUDA is accepted only after actual decoding; unavailable codecs/profiles reopen
  in software and resume without duplicate timestamps. Software orientation and
  source packet keyframe/duration metadata compensate for PyAV 17 CUDA transfer
  omitting frame properties. Source color interpretation is restored before RGB conversion.
- `media/timestamps.py`: integer ticks and Fraction arithmetic; nearest rounding
  ties away from zero; float time bases, missing PTS and non-increasing PTS are rejected.
- `media/frame_map.py`: exact CFR sample-and-hold grid, two-frame lookahead,
  and a record of every kept, duplicated and dropped source index, with rational
  output timing. Metadata grows with video length; decoded frame queues do not.
- `media/audio.py`: every audio track is planned explicitly. Compatible codecs
  with verified packet trim boundaries copy their packets. Otherwise AAC (at
  least 320 kbit/s, scaled for channel count) or explicitly selected PCM is encoded.
  Sample trims use integer arithmetic and preserve language/layout/disposition.
  An audio-drop request with audio present is rejected. Silent inputs work normally.
- `media/video_writer.py`: source timing preserved by default; explicit CFR is
  optional. Validated NVENC or x264/x265 writes temporary media only under
  `roop_3d/outputs/`. Both MOV movie and track timescales are set explicitly to
  avoid edit-list rounding. Every final video frame is decoded and checked to
  within one source tick; every audio track must remain present and decodable.
  Then the output is renamed atomically and a manifest records source/output
  SHA-256, codecs, hardware evidence states, counts, audio decisions, trim/map,
  and local synthetic-media provenance. Cancellation/errors remove temporaries;
  failed manifest publication rolls back the newly published video.
- `media/service.py` and `media/schema.py`: the existing probe, preview, and face
  inspector use the new facilities; upright frames and packet-derived keyframes
  reach the existing real detector. Compatibility fields remain available, with
  additive exact timing/color/rotation/layout fields on probe responses.

## Verification

All commands ran from this checkout's `app/` with the dedicated interpreter.

| Exact command | Exit | Final result |
| --- | ---: | --- |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_stage05_media.py roop_3d/tests/test_stage02_media.py -q` | 0 | 43 passed, 16.76 s |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q` | 0 | 240 passed; 18 subtests passed; 0 failed/skipped/blocked; 39.63 s |
| `roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py --report` | 0 | 68,392 comprehensive and 2,647 historical checks matched; 0 violations |
| `git diff --check` | 0 | No whitespace errors |

The full suite includes **33 Stage 05 tests**, generated CFR/VFR/rotated/silent/
multi-audio fixtures, 30000/1001 timing, nonzero origins, actual x264/x265/NVENC
outputs, AAC packet-copy priming, payload hash equality, exact PCM trims,
AAC trim duration, CFR drops/duplicates, software fallback, input errors,
cancellation/handle release, and publication failure rollback. No external video
downloads, skipped core tests, or inference mocks were used. The one warning is
an existing FastAPI/Starlette TestClient deprecation.

The complete test-run history, including initial failures and corrections, appears
in `STAGE05_VALIDATION.json`. Regressions exposed CUDA metadata loss and MOV
edit-list rounding; assertions were retained and the implementation was corrected.

Cancellation uses in-process PyAV/FFmpeg libraries, with no FFmpeg subprocess to
leave behind. Tests close and rename the decoder input on Windows, check partial
output cleanup, and successfully retry the same output name.

## Hardware evidence

The generated H.264 VFR source coded at 320x180 with a 90-degree display rotation
and two PCM tracks completed using **NVDEC and h264_nvenc: VERIFIED_LOCAL**.
The output was upright 180x320; all eight PTS and both copied audio payloads
matched. Hardware and software source decoding matched timing, keyframe flags,
durations, and decoded colors within the tested numerical tolerance.

The 96x64 encode profile fell back to x264 after NVENC validation failed. FFV1
fell back from CUDA to software. These states are per-input/profile; they are
not blanket hardware support or performance claims. Separate 6 GB/12 GB inference
acceptance campaigns remain outside Stage 05.

## Limits and next gate

Output is limited to MP4/MOV/MKV, even-sized YUV420 geometry, and right-angle
source rotations for encoding. The reader also supports arbitrary-angle upright
rotation. Unknown final-frame duration cannot be guessed for CFR conversion.
Missing/non-increasing PTS fail explicitly. If an output muxer's precision cannot
meet the source tick tolerance (notably fine-tick input into MKV), publication
fails and the partial output is removed. Probe samples do not claim whole-file
CFR unless they reach EOF. Declared frame counts can be unknown. Seek-frame
indices count emissions in that iterator; exact PTS identify seek results.

The writer processes one video stream and every audio track; it does not preserve
additional video streams, subtitles, or attachments. Source/output audio timing
and track decisions are explicit in the manifest. Pure remux of the original video
is not provided: video is encoded so Stage 06 can supply transformed frames.

Immutable legacy source/model baselines were never replaced. The audit separately
classified 1,456 runtime files under its existing explicit policy; those are not
claimed byte-identical. The full audit completed successfully before this report.

Stage 05 is passed. Stage 06 awaits its numbered specification; no Stage 06
inference adapter was implemented or claimed here.

## Changed files

- `app/roop_3d/media/audio.py`
- `app/roop_3d/media/frame_map.py`
- `app/roop_3d/media/probe.py`
- `app/roop_3d/media/timestamps.py`
- `app/roop_3d/media/video_reader.py`
- `app/roop_3d/media/video_writer.py`
- `app/roop_3d/media/schema.py`
- `app/roop_3d/media/service.py`
- `app/roop_3d/tests/test_stage05_media.py`
- `app/roop_3d/audit/STAGE05_REPORT.md`
- `app/roop_3d/audit/STAGE05_VALIDATION.json`

Ignored local audit output: `app/roop_3d/audit/stage00_verification.json`.

## Reference documentation

Existing dependency evidence is the pinned `av==17.1.0` lock and installed PyAV
17.1.0 sources (`av/video/codeccontext.py`, `frame.py`, `stream.py`). In particular,
`_transfer_hwframe` transfers only PTS, motivating the packet metadata recovery.
The [official PyAV timing guide](https://pyav.org/docs/stable/api/time.html)
describes the distinct codec/stream time bases and rational tick arithmetic;
the [current guide](https://pyav.basswood.io/docs/stable/api/time.html) corroborates
the muxer time-base change contract. Implementation uses the installed pinned
API rather than assuming newer documentation matches version 17.
