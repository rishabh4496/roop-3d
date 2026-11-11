"""Compatibility alias: the Batch Matrix engine now lives in :mod:`roop_3d.queue.batch_matrix` (Stage 23).

The module object itself is aliased, so ``import roop_3d.batch_matrix`` and attribute patching keep working.
"""
import sys

from roop_3d.queue import batch_matrix as _batch_matrix

sys.modules[__name__] = _batch_matrix
