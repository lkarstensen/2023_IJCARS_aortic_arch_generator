from .aorticarcharteries import (
    aorta_generator,
    brachiocephalic_trunk_static,
    left_common_carotid,
    left_common_carotid_II,
    left_subclavian,
    left_subclavian_IV,
    left_subclavian_VI,
    right_common_carotid,
    right_common_carotid_V,
    right_common_carotid_VII,
    right_subclavian,
    right_subclavian_IV,
    right_subclavian_V,
    right_subclavian_VI,
    common_origin_VI,
)

from .util.voxelcube import (
    create_empty_voxel_cube_from_branches,
    get_surface_mesh,
    save_mesh,
)
from .branch import Branch

from enum import Enum
from typing import Dict
import numpy as np
from tempfile import gettempdir
import os


class ArchType(str, Enum):
    I = "I"
    II = "II"
    IV = "IV"
    Va = "Va"
    Vb = "Vb"
    VI = "VI"
    VII = "VII"


class AorticArch:
    def __init__(
        self,
        arch_type: ArchType = ArchType.I,
        seed: int = None,
        rotate_y_deg: float = 0.0,
        rotate_z_deg: float = 0.0,
        rotate_x_deg: float = 0.0,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
        scale_diameter: float = 1.0,
        omit_y_axis: bool = False,
    ) -> None:
        self.arch_type = arch_type
        self.seed = seed
        self.rotate_y_deg = rotate_y_deg
        self.rotate_z_deg = rotate_z_deg
        self.rotate_x_deg = rotate_x_deg
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.scale_diameter = scale_diameter
        self.omit_y_axis = omit_y_axis

        self._mesh_path = None
        self.branches: Dict[str, Branch] = {}
        self._create_vessel_tree()
        self._rotate(rotate_y_deg, rotate_z_deg, rotate_x_deg)
        self._scale(scale_x, scale_y, scale_z, scale_diameter)
        if self.omit_y_axis:
            self._to_2d(axis_to_remove="y")

    @property
    def mesh_path(self) -> str:
        if self._mesh_path == None:
            self._mesh_path = self.generate_temp_mesh(0.99)
        return self._mesh_path

    @property
    def coordinates_high(self) -> np.ndarray:
        branch_highs = [branch.high for branch in self.branches.values()]
        high = np.max(branch_highs, axis=0)
        return high

    @property
    def coordinates_low(self) -> np.ndarray:
        branch_lows = [branch.low for branch in self.branches.values()]
        low = np.min(branch_lows, axis=0)
        return low

    def generate_temp_mesh(self, decimate_factor=0.99) -> str:
        while True:
            pid = os.getpid()
            nr = int(os.times().elapsed)
            mesh_path = f"{gettempdir()}/aorticarch_{pid}-{nr}.obj"
            if not os.path.exists(mesh_path):
                try:
                    open(mesh_path, "x").close()
                except IOError:
                    continue
                break
        self.generate_mesh(mesh_path, decimate_factor)
        return mesh_path

    def generate_mesh(self, mesh_path: str, decimate_factor: 0.99) -> None:
        voxel_cube = create_empty_voxel_cube_from_branches(
            self.branches, [0.6, 0.6, 0.9]
        )
        for _ in range(5):
            voxel_cube.add_padding_layer_all_sides()

        for branch in self.branches.values():
            voxel_cube.mark_centerline_in_array(
                branch, marking_value=1, radius_padding=0
            )
        voxel_cube.gaussian_smooth(1)
        voxel_cube.gaussian_smooth(1)

        mesh = get_surface_mesh(voxel_cube)
        mesh = mesh.decimate(decimate_factor)
        save_mesh(mesh, mesh_path)

    def _create_vessel_tree(self) -> None:

        rng = np.random.default_rng(self.seed)
        n = rng.normal

        aorta_resolution = 2
        bct_resolution = 1
        aorta, _ = aorta_generator(aorta_resolution, rng)
        if self.arch_type == ArchType.I:

            distance_aorta_end_bct = n(36, 5)
            idx = int(np.round(distance_aorta_end_bct / aorta_resolution, 0))
            bct, bct_chs_points = brachiocephalic_trunk_static(
                aorta.coordinates[-idx], 1, rng
            )

            rsa, _ = right_subclavian(bct.coordinates[-1], bct_chs_points[-1], 1, rng)
            rcca, _ = right_common_carotid(bct.coordinates[-1], 1, rng)

            distance_bct_lcca = n(16, 4)
            idx += int(np.round(distance_bct_lcca / aorta_resolution, 0))
            lcca, _ = left_common_carotid(aorta.coordinates[-idx], 1, rng)

            distance_lcca_lsca = n(28, 3)
            idx += int(np.round(distance_lcca_lsca / aorta_resolution, 0))
            lsa, _ = left_subclavian(aorta.coordinates[-idx], 1, rng)
            arteries = {
                "aorta": aorta,
                "bct": bct,
                "rcca": rcca,
                "rsa": rsa,
                "lcca": lcca,
                "lsa": lsa,
            }

        elif self.arch_type == ArchType.II:
            distance_aorta_end_bct = n(36, 3)
            idx = int(np.round(distance_aorta_end_bct / aorta_resolution, 0))
            bct, bct_chs_points = brachiocephalic_trunk_static(
                aorta.coordinates[-idx], bct_resolution, rng
            )
            rsa, _ = right_subclavian(bct.coordinates[-1], bct_chs_points[-1], 1, rng)
            rcca, _ = right_common_carotid(bct.coordinates[-1], 1, rng)

            distance_aorta_lcca = n(bct.length * (2 / 3), bct.length * (3 / 10) / 3)
            distance_aorta_lcca = abs(distance_aorta_lcca)  # if distance < 0
            distance_aorta_lcca = min(
                bct.length, distance_aorta_lcca
            )  # if distance longer than bct
            lcca_idx = int(np.round(distance_aorta_lcca / bct_resolution, 0))
            lcca, _ = left_common_carotid_II(
                bct.coordinates[lcca_idx], resolution=1, rng=rng
            )

            distance_bct_lsca = n(41, 2.5)
            idx += int(np.round(distance_bct_lsca / aorta_resolution, 0))
            lsa, _ = left_subclavian(aorta.coordinates[-idx], 1, rng)
            arteries = {
                "aorta": aorta,
                "bct": bct,
                "rcca": rcca,
                "rsa": rsa,
                "lcca": lcca,
                "lsa": lsa,
            }

        elif self.arch_type == ArchType.IV:
            distance_aorta_end_rsca = n(42, 5)
            idx = int(np.round(distance_aorta_end_rsca / aorta_resolution, 0))
            rsa, _ = right_subclavian_IV(aorta.coordinates[-idx], 1, rng)

            distance_rsca_co = n(20, 4)
            idx += int(np.round(distance_rsca_co / aorta_resolution, 0))
            co, co_chs_points = common_origin_VI(
                aorta.coordinates[-idx], bct_resolution, rng
            )

            rcca, _ = right_common_carotid(co.coordinates[-1], 1, rng)
            lcca, _ = left_common_carotid_II(co.coordinates[-1], 1, rng)

            distance_bct_lsca = n(38, 3)
            idx += int(np.round(distance_bct_lsca / aorta_resolution, 0))
            lsa, _ = left_subclavian_IV(aorta.coordinates[-idx], 1, rng)
            arteries = {
                "aorta": aorta,
                "co": co,
                "rcca": rcca,
                "rsa": rsa,
                "lcca": lcca,
                "lsa": lsa,
            }

        elif self.arch_type == ArchType.Va:
            distance_aorta_end_bct = n(36, 3)
            idx = int(np.round(distance_aorta_end_bct / aorta_resolution, 0))
            bct, bct_chs_points = brachiocephalic_trunk_static(
                aorta.coordinates[-idx], 1, rng
            )

            rcca, _ = right_common_carotid_V(
                bct.coordinates[-1], bct_chs_points[-1], 1, rng
            )
            lcca, _ = left_common_carotid_II(bct.coordinates[-1], 1, rng)

            distance_bct_lsca = n(41, 2.5)
            idx += int(np.round(distance_bct_lsca / aorta_resolution, 0))
            lsa, _ = left_subclavian(aorta.coordinates[-idx], 1, rng)

            distance_lsca_rsca = n(20, 1)
            idx += int(np.round(distance_lsca_rsca / aorta_resolution, 0))
            rsa, _ = right_subclavian_V(aorta.coordinates[-idx], 1, rng)

            arteries = {
                "aorta": aorta,
                "bct": bct,
                "rcca": rcca,
                "rsa": rsa,
                "lcca": lcca,
                "lsa": lsa,
            }

        elif self.arch_type == ArchType.Vb:
            distance_aorta_end_rcca = n(50, 2.5)
            idx = int(np.round(distance_aorta_end_rcca / aorta_resolution, 0))
            rcca, _ = right_common_carotid_VII(aorta.coordinates[-idx], 1, rng)

            distance_rcca_lcca = n(22, 3)
            idx += int(np.round(distance_rcca_lcca / aorta_resolution, 0))
            lcca, _ = left_common_carotid(aorta.coordinates[-idx], 1, rng)

            distance_lcca_lsca = n(26, 2.5)
            idx += int(np.round(distance_lcca_lsca / aorta_resolution, 0))
            lsa, _ = left_subclavian(aorta.coordinates[-idx], 1, rng)

            distance_lsca_rsca = n(20, 1)
            idx += int(np.round(distance_lsca_rsca / aorta_resolution, 0))
            rsa, _ = right_subclavian_V(aorta.coordinates[-idx], 1, rng)
            arteries = {
                "aorta": aorta,
                "rcca": rcca,
                "rsa": rsa,
                "lcca": lcca,
                "lsa": lsa,
            }

        elif self.arch_type == ArchType.VI:
            distance_aorta_end_bct = n(36, 3)
            idx = int(np.round(distance_aorta_end_bct / aorta_resolution, 0))
            bct, bct_chs_points = brachiocephalic_trunk_static(
                aorta.coordinates[-idx], 1, rng
            )

            rcca, _ = right_common_carotid_V(
                bct.coordinates[-1], bct_chs_points[-1], 1, rng
            )
            lcca, _ = left_common_carotid_II(bct.coordinates[-1], 1, rng)

            distance_bct_co = n(50, 1.5)
            idx += int(np.round(distance_bct_co / aorta_resolution, 0))
            co, co_chs_points = common_origin_VI(aorta.coordinates[-idx], 1, rng)

            rsa, _ = right_subclavian_VI(co.coordinates[-1], co_chs_points[-1], 1, rng)
            lsa, _ = left_subclavian_VI(co.coordinates[-1], 1, rng)
            arteries = {
                "aorta": aorta,
                "bct": bct,
                "co": co,
                "rcca": rcca,
                "rsa": rsa,
                "lcca": lcca,
                "lsa": lsa,
            }

        elif self.arch_type == ArchType.VII:
            distance_aorta_end_rsca = n(38, 3.5)
            idx = int(np.round(distance_aorta_end_rsca / aorta_resolution, 0))
            rsa, _ = right_subclavian_IV(aorta.coordinates[-idx], 1, rng)

            distance_rsca_rcca = n(20, 3)
            idx += int(np.round(distance_rsca_rcca / aorta_resolution, 0))
            rcca, _ = right_common_carotid_VII(aorta.coordinates[-idx], 1, rng)

            distance_rcca_lcca = n(20, 3)
            idx += int(np.round(distance_rcca_lcca / aorta_resolution, 0))
            lcca, _ = left_common_carotid(aorta.coordinates[-idx], 1, rng)

            distance_lcca_lsca = n(27, 3)
            idx += int(np.round(distance_lcca_lsca / aorta_resolution, 0))
            lsa, _ = left_subclavian(aorta.coordinates[-idx], 1, rng)

            arteries = {
                "aorta": aorta,
                "rcca": rcca,
                "rsa": rsa,
                "lcca": lcca,
                "lsa": lsa,
            }
        else:
            raise ValueError(f"{self.arch_type=} not supportet.")
        self.branches = arteries

    def _scale(self, x: float, y: float, z: float, diameter: float) -> None:
        xyz_sclaing = np.array([x, y, z])
        for branch in self.branches.values():
            branch.radii *= diameter
            branch.coordinates *= xyz_sclaing

    def _to_2d(self, axis_to_remove: str, dummy_value: float = 0) -> None:
        if axis_to_remove not in ["x", "y", "z"]:
            raise ValueError(
                f"{self.__class__.__name__}.to_2d() {axis_to_remove =} has to be 'x', 'y' or 'z'"
            )
        convert = {"x": 0, "y": 1, "z": 2}
        axis_to_remove = convert[axis_to_remove]
        for branch in self.branches.values():
            branch.coordinates[:, axis_to_remove] = dummy_value

    def _rotate(
        self, rotate_y_deg: float, rotate_z_deg: float, rotate_x_deg: float
    ) -> None:
        for branch in self.branches.values():
            branch.coordinates = self._rotate_array(
                array=branch.coordinates,
                x_deg=rotate_x_deg,
                z_deg=rotate_z_deg,
                y_deg=rotate_y_deg,
            )

    def _rotate_array(
        self,
        array: np.ndarray,
        y_deg: float,
        z_deg: float,
        x_deg: float,
    ):
        y_rad = y_deg * np.pi / 180
        lao_rao_rad = z_deg * np.pi / 180
        cra_cau_rad = x_deg * np.pi / 180

        rotation_matrix_y = np.array(
            [
                [np.cos(y_rad), 0, np.sin(y_rad)],
                [0, 1, 0],
                [-np.sin(y_rad), 0, np.cos(y_rad)],
            ],
        )

        rotation_matrix_lao_rao = np.array(
            [
                [np.cos(lao_rao_rad), -np.sin(lao_rao_rad), 0],
                [np.sin(lao_rao_rad), np.cos(lao_rao_rad), 0],
                [0, 0, 1],
            ],
        )

        rotation_matrix_cra_cau = np.array(
            [
                [1, 0, 0],
                [0, np.cos(cra_cau_rad), -np.sin(cra_cau_rad)],
                [0, np.sin(cra_cau_rad), np.cos(cra_cau_rad)],
            ],
        )
        rotation_matrix = np.matmul(rotation_matrix_cra_cau, rotation_matrix_lao_rao)
        rotation_matrix = np.matmul(rotation_matrix, rotation_matrix_y)
        # transpose such that matrix multiplication works
        rotated_array = np.matmul(rotation_matrix, array.T).T
        return rotated_array
