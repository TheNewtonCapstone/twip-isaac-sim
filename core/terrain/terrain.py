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
        scale: list[int],
    ):
        self.stage = stage

        self.size = size
        self.position = position
        self.rotation = rotation
        self.scale = scale


class TerrainBuilder(ABC):
    def __init__(
        self,
        size: list[int] = [1, 1],
        position: list[int] = [0, 0, 0],
        rotation: list[int] = [0, 0, 0],
        scale: list[int] = [1, 1, 1],
        randomize: bool = False,
    ):
        self.size = size
        self.position = position
        self.rotation = rotation
        self.scale = scale

        self.randomize = randomize

    @abstractmethod
    def build(self, stage: Any) -> TerrainBuild:
        pass

    def _heightmap_to_mesh(self, heightmap):
        x = torch.linspace(0, (self.size[0] - 1) * self.scale[0], self.size[0])
        y = torch.linspace(0, (self.size[1] - 1) * self.scale[1], self.size[1])
        yy, xx = torch.meshgrid(y, x)

        vertices = torch.stack(
            [xx.flatten(), yy.flatten(), heightmap.flatten() * self.scale[2]], dim=1
        )
        triangles = torch.zeros((self.size[0] - 1) * (self.size[1] - 1) * 2, 3)

        for i in range(self.size[0] - 1):
            # indices for the 4 vertices of the square
            ind0 = torch.arange(0, self.size[1] - 1) + i * self.size[1]
            ind1 = ind0 + 1
            ind2 = ind0 + self.size[1]
            ind3 = ind2 + 1

            # there are 2 triangles per square
            # and self.size[1] - 1 squares per col
            start = i * (self.size[1] - 1) * 2
            end = start + (self.size[1] - 1) * 2

            # first set of triangles (top right)
            triangles[start:end:2, 0] = ind0
            triangles[start:end:2, 1] = ind1
            triangles[start:end:2, 2] = ind2

            # second set of triangles (bottom left)
            triangles[start + 1 : end : 2, 0] = ind1
            triangles[start + 1 : end : 2, 1] = ind3
            triangles[start + 1 : end : 2, 2] = ind2

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

        terrain = XFormPrim(
            prim_path=f"/terrain_{num_faces}",
            name="terrain",
            position=self.position,
            orientation=None,
        )

        UsdPhysics.CollisionAPI.Apply(terrain.prim)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(terrain.prim)
        physx_collision_api.GetContactOffsetAttr().Set(0.02)
        physx_collision_api.GetRestOffsetAttr().Set(0.02)
