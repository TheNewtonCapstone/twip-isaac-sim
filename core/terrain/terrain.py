from abc import abstractmethod

import torch


class TerrainBuild:
    def __init__(
        self,
        stage,
        size: list[float],
        resolution: list[int],
        height: float,
        position: list[float],
        path: str
    ):
        self.stage = stage
        self.path = path

        self.size = size
        self.position = position
        self.resolution = resolution
        self.height = height


class TerrainBuilder:
    def __init__(
        self,
        size: list[float] = None,
        resolution: list[int] = None,
        height: float = 1,
        base_path: str = None,
    ):
        if size is None:
            size = [5, 5]
        if resolution is None:
            resolution = [10, 10]
        if base_path is None:
            base_path = "/World/terrains"

        self.size = size
        self.resolution = resolution
        self.height = height
        self.base_path = base_path

    def build_from_self(self, stage, position: list[float]) -> TerrainBuild:
        return self.build(
            stage,
            self.size,
            self.resolution,
            self.height,
            position,
            self.base_path
        )

    @staticmethod
    def build(
        stage,
        size: list[float],
        resolution: list[int],
        height: float,
        position: list[float],
        path="/World/terrains",
    ) -> TerrainBuild:
        """
        Builds a terrain in the stage, according to the class's implementation.

        Args:
            stage: USD stage to build the terrain in.
            size: Size of the terrain in the stage's units.
            resolution: Number of vertices per terrain.
            height: Height of the terrain in the stage's units.
            position: Position of the terrain in the stage's units.
            path: Path to the terrain in the stage.
        """
        pass

    @staticmethod
    def _add_heightmap_to_world(
        heightmap: torch.Tensor,
        size: list[float],
        num_cols: int,
        num_rows: int,
        height: float,
        base_path: str,
        builder_name: str,
        position: list[float]
    ) -> str:
        vertices, triangles = TerrainBuilder._heightmap_to_mesh(heightmap, size, num_cols, num_rows, height)

        return TerrainBuilder._add_mesh_to_world(vertices, triangles, base_path, builder_name, size, position)

    @staticmethod
    def _heightmap_to_mesh(
        heightmap: torch.Tensor,
        size: list[float],
        num_cols: int,
        num_rows: int,
        height: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # from https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/main/omniisaacgymenvs/utils/terrain_utils/terrain_utils.py

        x = torch.linspace(0, (size[0]), num_cols)
        y = torch.linspace(0, (size[1]), num_rows)
        xx, yy = torch.meshgrid(x, y)

        vertices = torch.zeros((num_cols * num_rows, 3), dtype=torch.float32)
        vertices[:, 0] = xx.flatten()
        vertices[:, 1] = yy.flatten()
        vertices[:, 2] = heightmap.flatten() * height
        triangles = torch.ones(
            (2 * (num_rows - 1) * (num_cols - 1), 3), dtype=torch.int32
        )
        for i in range(num_cols - 1):
            # indices for the 4 vertices of the square
            ind0 = torch.arange(0, num_rows - 1) + i * num_rows
            ind1 = ind0 + 1
            ind2 = ind0 + num_rows
            ind3 = ind2 + 1

            # there are 2 triangles per square
            # and self.size[1] - 1 squares per col
            start = i * (num_rows - 1) * 2
            end = start + (num_rows - 1) * 2

            # first set of triangles (top left)
            triangles[start:end:2, 0] = ind0
            triangles[start:end:2, 1] = ind3
            triangles[start:end:2, 2] = ind1

            # second set of triangles (bottom right)
            triangles[start + 1: end: 2, 0] = ind0
            triangles[start + 1: end: 2, 1] = ind2
            triangles[start + 1: end: 2, 2] = ind3

        return vertices, triangles

    @staticmethod
    def _add_mesh_to_world(
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        base_path: str,
        builder_name: str,
        size: list[float],
        position: list[float],
    ) -> str:
        from core.utils.usd import find_matching_prims
        from omni.isaac.core.prims.xform_prim import XFormPrim
        from omni.isaac.core.utils.prims import define_prim
        from pxr import UsdPhysics, PhysxSchema

        # generate an informative and unique name from the type of builder
        prim_path_expr = f"{base_path}/{builder_name}/terrain_.*"
        num_of_existing_terrains = len(find_matching_prims(prim_path_expr))
        prim_path = f"{base_path}/{builder_name}/terrain_{num_of_existing_terrains}"

        num_faces = triangles.shape[0]
        mesh = define_prim(prim_path, "Mesh")
        mesh.GetAttribute("points").Set(vertices.numpy())
        mesh.GetAttribute("faceVertexIndices").Set(triangles.flatten().numpy())
        mesh.GetAttribute("faceVertexCounts").Set([3] * num_faces)

        centered_position = [
            position[0] - size[0] / 2,
            position[1] - size[1] / 2,
            position[2],
        ]

        terrain = XFormPrim(
            prim_path=prim_path,
            name="terrain",
            position=centered_position,
        )

        UsdPhysics.CollisionAPI.Apply(terrain.prim)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(terrain.prim)
        physx_collision_api.GetContactOffsetAttr().Set(0.02)
        physx_collision_api.GetRestOffsetAttr().Set(0.02)

        return prim_path
