from dataclasses import dataclass
import numpy as np


@dataclass
class Branch:
    name: str
    coordinates: np.ndarray
    radii: np.ndarray

    @property
    def length(self) -> float:
        return np.sum(
            np.linalg.norm(self.coordinates[:-1] - self.coordinates[1:], axis=1)
        )

    @property
    def high(self) -> np.ndarray:
        shape = self.coordinates.shape
        radii = np.broadcast_to(self.radii.reshape((-1, 1)), shape)
        coords_high = self.coordinates + radii
        high_branch = np.max(coords_high, axis=0)
        return high_branch

    @property
    def low(self) -> np.ndarray:
        shape = self.coordinates.shape
        radii = np.broadcast_to(self.radii.reshape((-1, 1)), shape)
        coords_low = self.coordinates - radii
        low_branch = np.min(coords_low, axis=0)
        return low_branch

    def __repr__(self) -> str:
        return self.name
