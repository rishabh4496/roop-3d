"""Trajectory-local velocity adaptive filtering. Times are seconds, angles degrees."""
from __future__ import annotations
import numpy as np

from roop_3d.temporal import OneEuroFilter

class PoseHysteresis:
    def __init__(self, deadband=2.5):
        if not np.isfinite(deadband) or not 0 <= deadband < 10:
            raise ValueError("Invalid pose deadband")
        self.deadband, self.cell = deadband, None

    def update(self, yaw, pitch):
        if not np.isfinite([yaw, pitch]).all():
            raise ValueError("Pose must be finite")
        values = (yaw, pitch)
        if self.cell is None:
            self.cell = tuple(-1 if v < -e else 1 if v > e else 0 for v,e in zip(values,(15,10)))
        else:
            result=[]
            for v,e,old in zip(values,(15,10),self.cell):
                if old == -1 and v < -e+self.deadband: new=-1
                elif old == 1 and v > e-self.deadband: new=1
                else: new=-1 if v < -e-self.deadband else 1 if v > e+self.deadband else 0
                result.append(new)
            self.cell=tuple(result)
        return self.cell

class TemporalStabilizer:
    def __init__(self, **config):
        self.config, self.tracks = config, {}

    def _config(self):
        # Historic keyword ``d_cutoff`` is the canonical filter's ``derivative_cutoff``.
        return {("derivative_cutoff" if key == "d_cutoff" else key): value for key, value in self.config.items()}

    def reset(self):
        self.tracks.clear()

    def filter(self, track_id, channel, value, timestamp):
        key=(track_id,channel)
        state=self.tracks.get(key)
        # Discontinuities and seeks must never carry stale trajectory state.
        if state is None or timestamp <= state.time or timestamp-state.time > .5:
            state=self.tracks[key]=OneEuroFilter(**self._config())
        return state.update(value,timestamp)

    def retain(self, active_ids):
        self.tracks={key: value for key,value in self.tracks.items() if key[0] in active_ids}
