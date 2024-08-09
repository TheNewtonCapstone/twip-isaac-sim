from core.terrain.terrain import TerrainBuild, TerrainBuilder
import torch


class FlatTerrainBuild(TerrainBuild):
    def __init__(
        self,
        container,
        size: list[int],
        position: list[int],
        rotation: list[int],
        detail: list[int],
        height: float,
    ):
        super().__init__(container, size, position, rotation, detail)


# detail does not affect the flat terrain, the number of vertices is determined by the size
class FlatTerrainBuilder(TerrainBuilder):
    def build(self, stage):
        import omni.isaac.core
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr.Usd import Stage

        assert isinstance(stage, Stage), "Container must be of type UsdStage."

        heightmap = torch.zeros(self.size)

        vertices, triangles = self._heightmap_to_mesh(
            heightmap,
            num_cols=self.size[0],
            num_rows=self.size[1],
        )
        self._add_mesh_to_world(stage, vertices, triangles)

        return FlatTerrainBuild(
            stage,
            self.size,
            self.position,
            self.rotation,
            self.detail,
        )
