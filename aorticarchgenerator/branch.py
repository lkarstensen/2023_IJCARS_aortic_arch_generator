from dataclasses import dataclass
import numpy as np


@dataclass
class Branch:
    name: str
    cl_coordinates: np.ndarray
    cl_radii: np.ndarray

    @property
    def length(self) -> float:
        return np.sum(
            np.linalg.norm(self.cl_coordinates[:-1] - self.cl_coordinates[1:], axis=1)
        )

    @property
    def coordinates_high(self) -> np.ndarray:
        shape = self.cl_coordinates.shape
        radii = np.broadcast_to(self.cl_radii.reshape((-1, 1)), shape)
        coords_high = self.cl_coordinates + radii
        high_branch = np.max(coords_high, axis=0)
        return high_branch

    @property
    def coordinates_low(self) -> np.ndarray:
        shape = self.cl_coordinates.shape
        radii = np.broadcast_to(self.cl_radii.reshape((-1, 1)), shape)
        coords_low = self.cl_coordinates - radii
        low_branch = np.min(coords_low, axis=0)
        return low_branch

    def __repr__(self) -> str:
        return self.name
