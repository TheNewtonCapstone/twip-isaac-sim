from core.terrain.terrain import TerrainBuild, TerrainBuilder
import torch


class FlatTerrainBuild(TerrainBuild):
    def __init__(
        self,
        stage,
        prim_path: str,
        size: list[int],
        position: list[int],
        rotation: list[int],
        detail: list[int],
    ):
        super().__init__(stage, prim_path, size, position, rotation, detail, 0)


# detail does not affect the flat terrain, the number of vertices is determined by the size
class FlatTerrainBuilder(TerrainBuilder):
    def build(self, stage):
        import omni.isaac.core
        from pxr.Usd import Stage

        assert isinstance(stage, Stage), "Container must be of type UsdStage."

        heightmap = torch.zeros(self.size)

        terrain_path = self._add_heightmap_to_world(heightmap, self.size[0], self.size[1])

        return FlatTerrainBuild(
            stage,
            terrain_path,
            self.size,
            self.position,
            self.rotation,
            self.detail,
        )
