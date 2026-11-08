"""Measured identity constraint for aggressive photometric edits.

This constrains identity retention relative to the enhanced head, not the
accuracy of the upstream identity swap. Uses real verified ArcFace inference;
neither image correlation nor a synthetic descriptor substitutes for ArcFace.
"""
from dataclasses import dataclass, asdict

import numpy as np

from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class IdentityConstraint:
    requested_cosine: float
    accepted_cosine: float
    applied_fraction: float
    floor: float

    def metadata(self):
        return asdict(self)


def constrain_identity(before, candidate, landmarks, recognizer, *, floor=.8):
    """Choose the strongest tested blend that meets the cosine floor.

    The full physical result is evaluated first. Five bounded search steps
    select a verified result; no monotonicity assumption is used to accept an
    untested image. Rejected strength is reported rather than silently claimed
    as full-strength scene illumination. All candidates retain native resolution.
    """
    if not .78 < floor < 1:
        raise ValueError('Photometric identity floor must exceed .78 and be below 1')
    reference = recognizer.embed(before,landmarks)
    def cosine(image):
        value = float(reference @ recognizer.embed(image,landmarks))
        if not np.isfinite(value):
            raise DomainError('PHOTOMETRIC_IDENTITY_INVALID','ArcFace returned non-finite similarity',503)
        return value
    requested = cosine(candidate)
    if requested >= floor:
        return candidate, IdentityConstraint(requested,requested,1.,floor)
    accepted = cosine(before)
    if accepted < floor:
        raise DomainError('PHOTOMETRIC_IDENTITY_INVALID','ArcFace could not validate the enhanced baseline',503)
    result, low, high = before, 0., 1.
    base, delta = before.astype(np.float32), candidate.astype(np.float32)-before
    for _ in range(5):
        weight = (low+high)/2
        image = np.clip(base + delta*weight,0,255).round().astype(np.uint8)
        similarity = cosine(image)
        if similarity >= floor:
            result, accepted, low = image, similarity, weight
        else:
            high = weight
    return result, IdentityConstraint(requested,accepted,low,floor)
