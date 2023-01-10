from dataclasses import dataclass

from ..branch import Branch

from typing import Dict, List, Tuple


import numpy as np
import pyvista as pv


@dataclass
class VoxelCube:
    value_array: np.ndarray
    spacing: np.ndarray
    world_offset: np.ndarray

    def add_padding_layer(
        self, dimension: str, dimension_end: str, padding_value: float = 0
    ):
        assert dimension in ["x", "y", "z"], "dimension needs to be x,y or z"
        assert dimension_end in [
            "low",
            "high",
        ], "dimension_end needs to be 'low' or 'high'"

        if dimension == "x":
            add = np.ones((1, self.value_array.shape[1], self.value_array.shape[2]))
            axis = 0

        elif dimension == "y":
            add = np.ones((self.value_array.shape[0], 1, self.value_array.shape[2]))
            axis = 1

        else:
            add = np.ones((self.value_array.shape[0], self.value_array.shape[1], 1))
            axis = 2

        add *= padding_value
        if dimension_end == "low":
            self.value_array = np.concatenate([add, self.value_array], axis=axis)
            self.world_offset[axis] -= self.spacing[axis]
        else:
            self.value_array = np.concatenate([self.value_array, add], axis=axis)

    def add_padding_layer_all_sides(self, padding_value: float = 0):

        self.add_padding_layer(
            dimension="x", dimension_end="low", padding_value=padding_value
        )
        self.add_padding_layer(
            dimension="x", dimension_end="high", padding_value=padding_value
        )
        self.add_padding_layer(
            dimension="y", dimension_end="low", padding_value=padding_value
        )
        self.add_padding_layer(
            dimension="y", dimension_end="high", padding_value=padding_value
        )
        self.add_padding_layer(
            dimension="z", dimension_end="low", padding_value=padding_value
        )
        self.add_padding_layer(
            dimension="z", dimension_end="high", padding_value=padding_value
        )

    def remove_layer(self, dimension: str, dimension_end: str):
        assert dimension in ["x", "y", "z"], "dimension needs to be x,y or z"
        assert dimension_end in [
            "low",
            "high",
        ], "dimension_end needs to be 'low' or 'high'"
        if dimension == "x":
            axis = 0

        elif dimension == "y":
            axis = 1

        else:
            axis = 2

        if dimension_end == "low":
            self.value_array = np.delete(self.value_array, 0, axis)
            self.world_offset[axis] += self.spacing[axis]
        else:
            self.value_array = np.delete(self.value_array, -1, axis)

    def gaussian_smooth(self, sigma: float):
        from skimage.filters import gaussian

        self.value_array = gaussian(self.value_array, sigma)

    def mark_centerline_in_array(
        self,
        branch: Branch,
        marking_value: float,
        radius_padding: float,
    ):

        for coordinates, radius in zip(branch.cl_coordinates, branch.cl_radii):
            (
                voxel_idxs_of_interest,
                voxels_of_interest,
            ) = self._get_voxel_idxs_of_interest(coordinates, radius + radius_padding)
            mask = self._in_sphere(
                coordinates, radius + radius_padding, voxels_of_interest
            )
            idxs_to_mark = voxel_idxs_of_interest[np.where(mask)[0]]
            self.value_array[
                idxs_to_mark[:, 0], idxs_to_mark[:, 1], idxs_to_mark[:, 2]
            ] = marking_value

    def _get_voxel_idxs_of_interest(
        self, coordinates: np.ndarray, radius: float
    ) -> np.ndarray:
        coords_min = coordinates - radius
        coords_max = coordinates + radius

        voxel_array_min_idx = np.round(
            (coords_min - self.world_offset) / self.spacing, 0
        ).astype(np.int16)
        voxel_array_max_idx = np.round(
            (coords_max - self.world_offset) / self.spacing, 0
        ).astype(np.int16)

        shape = (voxel_array_max_idx + 1) - voxel_array_min_idx

        idx_x_base_array = np.arange(
            voxel_array_min_idx[0], voxel_array_max_idx[0] + 1, dtype=np.int16
        ).reshape(-1, 1, 1)
        idx_y_base_array = np.arange(
            voxel_array_min_idx[1], voxel_array_max_idx[1] + 1, dtype=np.int16
        ).reshape(1, -1, 1)
        idx_z_base_array = np.arange(
            voxel_array_min_idx[2], voxel_array_max_idx[2] + 1, dtype=np.int16
        ).reshape(1, 1, -1)
        pos_x_base_array = idx_x_base_array * self.spacing[0] + self.world_offset[0]
        pos_y_base_array = idx_y_base_array * self.spacing[1] + self.world_offset[1]
        pos_z_base_array = idx_z_base_array * self.spacing[2] + self.world_offset[2]

        idx_x_array = np.broadcast_to(idx_x_base_array, shape)
        idx_y_array = np.broadcast_to(idx_y_base_array, shape)
        idx_z_array = np.broadcast_to(idx_z_base_array, shape)

        voxel_idxs_of_interest = np.stack(
            (idx_x_array, idx_y_array, idx_z_array), axis=-1
        ).reshape(-1, 3)

        pos_x_array = np.broadcast_to(pos_x_base_array, shape)
        pos_y_array = np.broadcast_to(pos_y_base_array, shape)
        pos_z_array = np.broadcast_to(pos_z_base_array, shape)

        voxels_of_interest = np.stack(
            (pos_x_array, pos_y_array, pos_z_array), axis=-1
        ).reshape(-1, 3)

        return voxel_idxs_of_interest, voxels_of_interest

    @staticmethod
    def _in_sphere(coordinates: np.ndarray, radius: float, voxels: np.ndarray):
        distances = np.linalg.norm(voxels - coordinates, axis=1)
        return np.less(distances, radius)


def create_empty_voxel_cube_from_branches(
    branches: Dict[str, Branch], spacing: Tuple[float, float, float]
) -> VoxelCube:
    spacing = np.array(spacing, dtype=np.float32)

    branch_lows = [branch.coordinates_low for branch in branches.values()]
    low = np.min(branch_lows, axis=0)

    branch_highs = [branch.coordinates_high for branch in branches.values()]
    high = np.max(branch_highs, axis=0)

    axes_length = high - low
    world_offset = low

    shape = np.ceil(axes_length / spacing).astype(int)

    shape += 2
    world_offset -= spacing
    voxel_array = np.zeros(shape, dtype=np.float32)
    voxel_cube = VoxelCube(voxel_array, spacing, world_offset)

    return voxel_cube


def get_surface_mesh(voxel_cube: VoxelCube) -> pv.PolyData:
    from skimage import measure

    vertices, faces, _, _ = measure.marching_cubes(
        voxel_cube.value_array,
        None,
        spacing=voxel_cube.spacing,
        gradient_direction="descent",
    )
    faces_nr_points = np.ones((faces.shape[0], 1), dtype=np.int64)
    faces_nr_points *= 3
    faces = np.concatenate((faces_nr_points, faces), axis=1)
    mesh = pv.PolyData(vertices, faces)
    mesh.translate(voxel_cube.world_offset, inplace=True)
    return mesh


def save_mesh(mesh: pv.PolyData, path: str):
    pv.save_meshio(path, mesh)
