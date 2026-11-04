# Performance guide

Roop 3D does not promise a speed: it depends on your GPU, your video and what you switch on. This guide explains what costs time, what is cheap to
change, and how to measure. No figure below is a benchmark; where a number appears it is a real default or limit.

## 1. What decides how long a render takes

1. **Where it runs**: TensorRT is fastest, CUDA next, CPU far behind ([GPU troubleshooting](GPU_TROUBLESHOOTING.md#2-hardware-profiles)).
2. **How many frames**: only the range between **In** and **Out** is rendered. Trimming to the part you need is the biggest saving.
3. **Resolution and number of faces** in the target.
4. **What you switch on**: a restoration model, a 3D render mode, relighting, a high-quality export preset all add work per frame.
5. **Whether the whole video must be analysed**: see the three stages below.

## 2. The three stages

The run strip shows them while a job runs:

| Stage | What happens |
|---|---|
| **Step 1 · Pre-Scan** | A sparse scan (every `sparse_interval` frames, default **25**) estimates where faces are |
| **Step 2 · Range Lock** | Frame ranges that contain faces are locked for processing; empty stretches can be skipped (*Skipped* shows the share) |
| **Step 3 · Render & Mux** | Frames are swapped, composited, encoded and combined with the audio |

The streaming backend reports all three; the reference backend skips steps 1 and 2 and shows them as *skipped*. The strip also shows **FPS** and an
**ETA** once they are measured; unmeasured values stay as "—" rather than a guess.

## 3. Cheap wins, in order

| Change | Effect | Cost |
|---|---|---|
| Trim to the part you need (**[ In** / **Out ]**) | Renders fewer frames | None |
| Export preset **Standard MP4** | The only preset allowed to use hardware (NVENC) encoding; the others use slower software settings | Less quality headroom than *High-quality MP4* |
| Restoration **off** (default) | Skips a heavy model per face per frame | Softer faces |
| **Standard identity swap** | Cheaper than the 3D render modes | No 3D turning |
| Close other GPU programs | More free video memory → the `12gb` profile and larger batches | None |
| Turn **Live swap** off while you are not judging the swap | The preview stops computing swaps | You see the original frame |
| Keep Roop 3D running between jobs | Warm models and cached engines are reused | None |

## 4. Settings that matter (Settings → Workstation Preferences)

| Tab | Setting (default) | What it does |
|---|---|---|
| Compute | *Hardware profile* (`auto`) · *Preferred provider* (`auto`) · *Precision policy* (`auto`) | See [Engine comparison §4](ENGINE_COMPARISON.md) |
| Compute | *VRAM limit mb* (0 = none) · *Memory safety reserve mb* (768) | Caps GPU memory use; keeps a reserve free for the system |
| Compute | *Execution threads* (1) · *Stream batch size* (4) · *Stream ring slots* (4) | CPU threads and how many frames are in flight in the streaming backend. Larger batches need more memory |
| Pipeline | *Face restoration policy* (`off`) · *Restorer threads* (1) · *Restorer pool size* (2) | Restoration and how many restorers run in parallel |
| Pipeline | *Sparse interval* (25) · *Stream segment frames* (250) | Pre-scan density, and the segment length for streaming |
| UX & Diagnostics | *Max concurrent jobs* (1) | How many renders run at once. Raise only with ample memory |
| Finishing | *Video timing policy* (`preserve`) · *Audio policy* (`copy`) | Keep the source frame timing and audio |

Change one thing at a time and watch the FPS in the run strip.

## 5. Preview versus render

The live preview and a render use the same GPU. While a render, a model build or a TensorRT compile owns it the preview reports *busy* and retries
by itself, and queued previews are dropped when a render starts so it can take the GPU promptly. Previews are cached by their settings, so asking
for the same frame again is instant. The preview always requests the **full-resolution frame**: shrinking the window changes only the space the
picture occupies, never the resolution the server produces.

## 6. Checkpoints, pausing and resuming

* A checkpoint of finished output frames is written about every 30 frames.
* Resume replays the analysis from the first frame and skips only re-saving finished frames, so **a resumed render is not faster than starting
  over for the analysis part**. A render that stopped at minute 55 of 60 needs about 55 minutes to get back. Pause only when you must.
* Long renders can be stopped by security software that dislikes sustained disk activity; the job returns as *Interrupted* and can be resumed.
  If it happens repeatedly, look at your security software's log for the event; whether to add an exclusion is your decision.
* Finished frames are released when a job fails, so a failed job does not keep tens of gigabytes.

## 7. Measuring

* **Run strip**: stage, percentage, skipped %, FPS, ETA.
* **Processing screen**: per-stage timings, the **bottleneck flamegraph** and a GPU readout with **Purge VRAM** (unload idle cached models).
* **Header pill**: memory use and whether the GPU is busy.
* **Settings → Diagnostics Suite**: a 14-section report.

## 8. Disk

Imported videos are full copies in `app/roop_3d/.runtime/media`. Outputs go to `app/roop_3d/.runtime/outputs`. Keep several times the length of your video
free, and use a fast drive: a render refuses to start if space is low (`DISK_SPACE_LOW`) and fails visibly if the disk fills (`DISK_FULL`).
