# Roop 3D v0.2.0 Migration & Upgrade Guide

## 1. Overview

Roop 3D v0.2.0 introduces automated CI/CD verification gates, hardware profiles (`12gb`, `6gb`, `cpu`), and release manifests. This guide details how to safely upgrade from v0.1.x without data loss or downtime.

---

## 2. Safety Guarantees

Upgrading to v0.2.0 strictly preserves:
- All created **FaceSets** in `storage/facesets/`
- All saved **Projects** and presets in `storage/projects/`
- All rendered **Outputs** in `outputs/`
- All downloaded **Model Weights** in cache/registry
- Complete filesystem and process isolation from `roop-ultimate`

---

## 3. Recommended Upgrade Procedure

### Step 1: Pre-Upgrade Configuration Backup
Before initiating the update, create a backup of your current configuration using the admin CLI:

```bash
app/roop_3d/.venv/Scripts/python.exe tools/project_admin.py backup-config
```
This stores an atomic snapshot in `cache/backups/`.

### Step 2: Pull Latest Release
Inside the Roop 3D directory, pull the latest release commits:

```bash
git fetch origin main
git pull origin main
```

Or trigger the update via the Pinokio UI by clicking **Update** (`update.js`).

### Step 3: Run Environment Preflight & Verification
Verify your local runtime against the 12-point preflight check:

```bash
app/roop_3d/.venv/Scripts/python.exe tools/project_admin.py preflight
```

### Step 4: Verify Release Integrity & CI Gates
Ensure your local checkout passes the verification suite:

```bash
app/roop_3d/.venv/Scripts/python.exe tools/ci_runner.py --gate 1
app/roop_3d/.venv/Scripts/python.exe tools/ci_runner.py --gate 13
```

---

## 4. Rollback in Case of Failure

If an update is interrupted or causes unexpected issues, execute a rollback to the saved backup:

```bash
app/roop_3d/.venv/Scripts/python.exe tools/project_admin.py rollback
```

Your previous configuration will be restored immediately.
