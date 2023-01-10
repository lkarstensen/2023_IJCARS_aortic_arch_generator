from aorticarchgenerator import AorticArch, ArchType
import os

for type in [
    ArchType.I,
    ArchType.II,
    ArchType.IV,
    ArchType.Va,
    ArchType.Vb,
    ArchType.VI,
    ArchType.VII,
]:

    vessels = AorticArch(
        arch_type=type,
        seed=1,
        scale_x=1.0,
        scale_y=1.0,
        scale_z=1.0,
        scale_diameter=1.0,
        omit_y_axis=False,
    )
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mesh_folder = os.path.join(dir_path, "meshes")
    if not os.path.exists(mesh_folder):
        os.mkdir(mesh_folder)

    mesh_path = os.path.join(mesh_folder, f"aortic_arch_type_{type.value}.obj")
    vessels.create_mesh(mesh_path=mesh_path, decimate_factor=0.8)
