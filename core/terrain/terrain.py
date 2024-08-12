from abc import abstractmethod

import torch


class TerrainBuild:
    def __init__(
        self,
        stage,
        path: str,
        size: list[int],
        position: list[int],
        rotation: list[int],
        detail: list[int],
        height: float,
    ):
        self.stage = stage
        self.path = path

        self.size = size
        self.position = position
        self.rotation = rotation
        self.detail = detail
        self.height = height


class TerrainBuilder:
    def __init__(
        self,
        base_path: str = None,
        size: list[int] = None,
        position: list[int] = None,
        rotation: list[int] = None,
        detail: list[int] = None,
        height: float = 1,
    ):
        if base_path is None:
            base_path = "/World/terrains"
        if detail is None:
            detail = [1, 1]
        if rotation is None:
            rotation = [0, 0, 0]
        if position is None:
            position = [0, 0, 0]
        if size is None:
            size = [1, 1]

        self.base_path = base_path
        self.size = size
        self.position = position
        self.rotation = rotation
        self.detail = detail
        self.height = height

    def build(self, base_terrain_path: str) -> TerrainBuild:
        pass

    def _add_heightmap_to_world(self, heightmap: torch.Tensor, num_cols: int, num_rows: int) -> str:
        vertices, triangles = self._heightmap_to_mesh(heightmap, num_cols, num_rows)

        return self._add_mesh_to_world(vertices, triangles)

    def _heightmap_to_mesh(self, heightmap: torch.Tensor, num_cols: int, num_rows: int):
        # from https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/main/omniisaacgymenvs/utils/terrain_utils/terrain_utils.py

        x = torch.linspace(0, (self.size[0]), num_cols)
        y = torch.linspace(0, (self.size[1]), num_rows)
        xx, yy = torch.meshgrid(x, y)

        vertices = torch.zeros((num_cols * num_rows, 3), dtype=torch.float32)
        vertices[:, 0] = xx.flatten()
        vertices[:, 1] = yy.flatten()
        vertices[:, 2] = heightmap.flatten() * self.height
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

    def _add_mesh_to_world(
        self, vertices: torch.Tensor, triangles: torch.Tensor
    ) -> str:
        from core.utils.usd import find_matching_prims
        from omni.isaac.core.prims.xform_prim import XFormPrim
        from omni.isaac.core.utils.prims import define_prim
        from pxr import UsdPhysics, PhysxSchema

        # generate an informative and unique name from the type of builder
        clean_instance_name = self.__class__.__name__.replace("TerrainBuilder", "").lower()
        prim_path_expr = f"{self.base_path}/{clean_instance_name}/terrain_.*"
        num_of_existing_terrains = len(find_matching_prims(prim_path_expr))
        prim_path = f"{self.base_path}/{clean_instance_name}/terrain_{num_of_existing_terrains}"

        num_faces = triangles.shape[0]
        mesh = define_prim(prim_path, "Mesh")
        mesh.GetAttribute("points").Set(vertices.numpy())
        mesh.GetAttribute("faceVertexIndices").Set(triangles.flatten().numpy())
        mesh.GetAttribute("faceVertexCounts").Set([3] * num_faces)

        centered_position = [
            self.position[0] - self.size[0] / 2,
            self.position[1] - self.size[1] / 2,
            self.position[2],
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
