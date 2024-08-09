from abc import ABC, abstractmethod
from typing import Any

import torch


class TerrainBuild(ABC):
    def __init__(
        self,
        stage,
        size: list[int],
        position: list[int],
        rotation: list[int],
        detail: list[int],
        height: float,
    ):
        self.stage = stage

        self.size = size
        self.position = position
        self.rotation = rotation
        self.detail = detail
        self.height = height


class TerrainBuilder(ABC):
    def __init__(
        self,
        size: list[int] = [1, 1],
        position: list[int] = [0, 0, 0],
        rotation: list[int] = [0, 0, 0],
        detail: list[int] = [1, 1],
        height: float = 1,
        randomize: bool = False,
    ):
        self.size = size
        self.position = position
        self.rotation = rotation
        self.detail = detail
        self.height = height

        self.randomize = randomize

    @abstractmethod
    def build(self, stage: Any) -> TerrainBuild:
        pass

    def _heightmap_to_mesh(self, heightmap, num_cols, num_rows):
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
            triangles[start + 1 : end : 2, 0] = ind0
            triangles[start + 1 : end : 2, 1] = ind2
            triangles[start + 1 : end : 2, 2] = ind3

        return vertices, triangles

    def _add_mesh_to_world(
        self, stage, vertices: torch.Tensor, triangles: torch.Tensor
    ):
        import omni.isaac.core
        from omni.isaac.core.prims.xform_prim import XFormPrim
        from pxr import UsdPhysics, PhysxSchema

        num_faces = triangles.shape[0]
        mesh = stage.DefinePrim(f"/terrain_{num_faces}", "Mesh")
        mesh.GetAttribute("points").Set(vertices.numpy())
        mesh.GetAttribute("faceVertexIndices").Set(triangles.flatten().numpy())
        mesh.GetAttribute("faceVertexCounts").Set([3] * num_faces)

        centered_position = [
            self.position[0] - self.size[0] / 2,
            self.position[1] - self.size[1] / 2,
            self.position[2],
        ]

        terrain = XFormPrim(
            prim_path=f"/terrain_{num_faces}",
            name="terrain",
            position=centered_position,
            orientation=None,
        )

        UsdPhysics.CollisionAPI.Apply(terrain.prim)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(terrain.prim)
        physx_collision_api.GetContactOffsetAttr().Set(0.02)
        physx_collision_api.GetRestOffsetAttr().Set(0.02)
