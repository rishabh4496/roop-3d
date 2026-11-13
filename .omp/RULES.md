# ROOP-3D NON-NEGOTIABLE SWARM RULES

1. **Repository Independence:**
   - Backend Python code lives in `app/roop_3d/`. Frontend React/TS code lives in `app/roop_3d_ui/`.
   - `roop-ultimate` is a READ-ONLY external model store accessed solely via the hashed model registry. NEVER modify, write to, or delete any file in `roop-ultimate`.
   - Maintain offline capability and loopback security (`127.0.0.1:8030`).

2. **Zero Stub / Zero Silent Fallback Policy:**
   - NEVER write mock functions, placeholder 2D `cv2.warpAffine` crops pretending to be 3D mesh projections, or `try/except: return frame` blocks that silently bypass Ultramax Enhancer, XSeg, or Multiband Blending.
   - Every stage must execute real tensor/mesh math and raise structured diagnostics if weights or tensors are invalid.

3. **OMP Subagent Dispatch Law:**
   - When calling the `task` tool for custom agents (`a01-3d-geo-core` through `a06-qa-verifier`), NEVER pass the `model` field in task arguments; models and effort are bound by `task.agentModelOverrides` and agent frontmatter. Per-call model overrides can conflict with that routing.
   - Batch independent parallel stages in one `task` call. Report cross-module contract changes and blockers through task results to the coordinator; do not rely on an `irc` tool, which OMP does not provide as a built-in tool.
