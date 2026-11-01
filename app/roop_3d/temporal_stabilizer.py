"""Trajectory-local velocity adaptive filtering. Times are seconds, angles degrees."""
from __future__ import annotations
import numpy as np

class OneEuroFilter:
    def __init__(self, min_cutoff=0.15, beta=0.04, d_cutoff=1.0):
        if not np.isfinite([min_cutoff, beta, d_cutoff]).all() or min_cutoff <= 0 or beta < 0 or d_cutoff <= 0:
            raise ValueError("Invalid One-Euro configuration")
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.time = self.raw = self.value = self.derivative = None

    def __call__(self, value, timestamp):
        value = np.asarray(value, dtype=np.float64)
        if not np.isfinite(value).all() or not np.isfinite(timestamp):
            raise ValueError("Filter inputs must be finite")
        if self.time is None:
            self.time, self.raw, self.value = timestamp, value.copy(), value.copy()
            self.derivative = np.zeros_like(value)
            return value.copy()
        dt = timestamp - self.time
        if dt <= 0 or value.shape != self.raw.shape:
            raise ValueError("Filter needs increasing timestamps and stable shapes")
        def alpha(cutoff):
            return 1 / (1 + 1 / (2 * np.pi * cutoff * dt))
        a = alpha(self.d_cutoff)
        self.derivative = a * (value - self.raw) / dt + (1-a) * self.derivative
        a = alpha(self.min_cutoff + self.beta * np.abs(self.derivative))
        self.value = a * value + (1-a) * self.value
        self.time, self.raw = timestamp, value.copy()
        return self.value.copy()

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

    def reset(self):
        self.tracks.clear()

    def filter(self, track_id, channel, value, timestamp):
        key=(track_id,channel)
        state=self.tracks.get(key)
        # Discontinuities and seeks must never carry stale trajectory state.
        if state is None or timestamp <= state.time or timestamp-state.time > .5:
            state=self.tracks[key]=OneEuroFilter(**self.config)
        return state(value,timestamp)

    def retain(self, active_ids):
        self.tracks={key: value for key,value in self.tracks.items() if key[0] in active_ids}
