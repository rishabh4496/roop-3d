---
title: Data-kind vocabulary
label: wayfinder:grilling
status: closed
assignee: rishabh4496
blocked_by: [01-engine-inventory.md]
---

## Question

What is the closed vocabulary of data kinds that Engine `inputs`/`outputs` use (e.g. frame, face_boxes, landmarks5, landmarks106, embedding, aligned_crop, mesh, expression_params, occluder_mask, swapped_crop, encoded_stream), what shape/resolution/precision constraints can a kind carry, and how does plan validation treat a mismatch between neighbouring Engines (no auto-conversion, per the settled rule)? The inventory (ticket 01) supplies the real inputs and outputs the vocabulary must cover.

## Resolution

Resolved by delegation (defaults adopted, not individually confirmed). Closed vocabulary of 13 kinds in `schema.py`: `video_file`, `identity_source`, `frame`, `face_detections`, `face_embedding`, `landmarks_106`, `landmarks_203`, `swapped_crop`, `identity_asset_3d`, `visible_face_map`, `restored_crop`, `composited_frame`, `encoded_video`. Coarser than the inventory's 19 candidates: crop sizes, tensor shapes and BGR/RGB/CUDA variants are adapter-internal. `video_file` and `identity_source` come from the source; the built-in compositor derives `composited_frame` from `frame` plus `swapped_crop` or `restored_crop`. A plan must form a typed chain in category order; a missing input is `CONTRACT_MISMATCH`; nothing is auto-converted. Shape, resolution and precision constraints live on the Engine entry (resolution bounds, precisions), not on the kinds.
