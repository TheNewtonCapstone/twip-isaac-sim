from core.terrain.terrain import TerrainBuild, TerrainBuilder
import torch


class FlatTerrainBuild(TerrainBuild):
    def __init__(
        self,
        container,
        size: list[int],
        position: list[int],
        rotation: list[int],
        scale: list[int],
    ):
        super().__init__(container, size, position, rotation, scale)


class FlatTerrainBuilder(TerrainBuilder):
    def __init__(
        self,
        size: list[int] = [2, 2],
        position: list[int] = [0, 0, 0],
        rotation: list[int] = [0, 0, 0],
        scale: list[int] = [1, 1, 1],
        randomize: bool = False,
    ):
        super().__init__(size, position, rotation, scale, randomize)

    def build(self, stage):
        import omni.isaac.core
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr.Usd import Stage

        assert isinstance(stage, Stage), "Container must be of type UsdStage."

        heightmap = torch.zeros(self.size)

        vertices, triangles = self._heightmap_to_mesh(heightmap)
        self._add_mesh_to_world(stage, vertices, triangles)

        return FlatTerrainBuild(
            stage,
            self.size,
            self.position,
            self.rotation,
            self.scale,
        )
