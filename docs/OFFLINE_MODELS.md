# Offline model installation

Roop 3D never needs the internet to process a video, but it does need its **model files** on disk, and it checks every one of them against a
recorded SHA-256 hash and size before it will load it. This guide shows how to see what is missing, how to put model files in place without a
network, and what the safety checks mean.

## 1. See what is missing

* In the app: **Settings → Environment & Offline Manager → Offline Inference & Core Models → Verify Offline Health**.
* From a terminal (project folder): `python -B tools/project_admin.py offline-health`.

The core models Roop 3D needs to inspect faces and swap them are:

| Role | File name |
|---|---|
| Detector | `scrfd_10g_bnkps.onnx` |
| Recognizer | `w600k_r50.onnx` |
| Swapper | `inswapper_128.onnx` |

For each missing file the report prints the exact name, its registered ID, the **expected SHA-256 and size**, and where to put it. Optional
models (for example the face restorers `gpen_bfr_512.onnx` and `CodeFormerv0.1.onnx`, or the face-parsing and 3D models) are listed in
**Settings → Model Hub & TRT Engine Manager** with their status.

## 2. Get the files (on a computer with a network)

Download each file from a source you trust and check that its size and SHA-256 match the values the report gives. A file whose hash does not
match is **rejected**, whatever its name. Roop 3D does not download models by itself: the Model Hub downloads only when you press a download
button, accept the licence where one is required, and the transfer is hash-verified and cancellable (the hub's policy is
*offline by default, explicit user action, hash verified*).

## 3. Put them in place

You can use any one of these ways.

**A. Import a bundle (recommended).** Put the files in one folder, or zip the folder, then run, in the project folder:

```text
python -B tools/project_admin.py model-import --bundle <path-to-folder-or-zip>
python -B tools/project_admin.py model-verify
```

`model-import` copies each file whose name and hash match a registered model into `app/roop_3d/models/cache/<model id>.model.<ext>`.
The same import is available from the app's installer API (`POST /api/v1/installer/models/import`, `…/models/scan` to preview a folder,
`…/models/verify`).

**B. Copy by hand.** Rename the file to `<registered model id>.model.onnx` (the report prints the exact target) and place it in
`app/roop_3d/models/cache/`.

Afterwards press *Verify Offline Health* again. Each file shows `VERIFIED_LOCAL` once its hash matches; `MODEL_DRIFT` means the file on disk
differs from the registered hash (restore the original file); `NOT_INSTALLED` means it was not found.

### What a bundle may contain

To keep an imported archive from doing harm, Roop 3D checks every entry **before** extracting anything:

* only `.onnx`, `.bin`, `.json`, `.hash` and `.safetensors` files are accepted; executables and scripts (`.exe .bat .cmd .ps1 .sh .py .dll .so
  .dylib .vbs`) are refused;
* absolute paths, drive letters, `..` traversal, symbolic links and special files are refused;
* one file may be at most 5 GiB and the expanded archive at most 20 GiB;
* every model file's SHA-256 and size are verified before it is published into the cache; a corrupted file is rejected.

## 4. Model review: why a verified file can still be refused

Having the right bytes is not the whole check. Before a model may run, Roop 3D's governance gate requires that its **record is fully
reviewed**:

* dependency compatibility is known and satisfied;
* hardware requirements are known (and your GPU meets the minimum, if one is recorded);
* the artifact's security review is `VERIFIED` and it is not a pickle-based format;
* precision requirements are known;
* source repository, revision, architecture and attribution are recorded;
* the licence id and URL, commercial-use and redistribution terms are known.

If any of these is unknown the model's state is **`LICENSE_RESTRICTED`** (licence gaps) or **`INCOMPATIBLE`** (other gaps) and the operation
stops with **`LICENSE_UNVERIFIED`** or **`MODEL_INCOMPATIBLE`** and a list of exactly what is unknown. There is no setting that skips this check
and retrying does not change it.

> **Known limitation on the build this guide was written against (checked 2026-11-05).** The model registry shipped with this checkout records
> the licence, security, precision and provenance fields of the core models as `UNKNOWN`. On that build, **inspecting faces and rendering are
> refused with `LICENSE_UNVERIFIED` even when every model file is present and verified**, and there is no button in the interface that can
> clear it: the review records have to be completed in the registry by whoever maintains the installation. The interface reports this with a plain
> explanation (*"A model has not passed its licence and safety review"*). If your *Verify Offline Health* shows every model `VERIFIED_LOCAL`
> and inspection still fails with this code, this is the reason.

## 4a. Reading the Model Hub

**Settings → Model Hub & TRT Engine Manager** lists every registered model and engine with a status:

| Status | Meaning |
|---|---|
| `READY_FOR_INFERENCE` | Installed, hash-checked, reviewed, able to run |
| `AVAILABLE` | Known locally, not yet verified for inference |
| `LICENSE_RESTRICTED` | Present, but its licence/review record is incomplete or restrictive |
| `UNVERIFIED` · `INCOMPATIBLE` | No verified adapter, or it does not fit this pipeline |
| `NOT_INSTALLED` | The file is not on this computer |
| `BROKEN` | Present but failed verification |

## 5. Updating and backing up

`check-update`, `backup-config` and `rollback` (see the [installation guide](INSTALLATION_GUIDE.md#7-command-line-optional)) back up and restore
your *settings*; they do not move model files. Model files are never deleted by *Update* or *Reset*.
