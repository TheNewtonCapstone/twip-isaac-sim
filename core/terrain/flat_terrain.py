from core.terrain.terrain import TerrainBuild, TerrainBuilder
import torch


class FlatTerrainBuild(TerrainBuild):
    def __init__(
        self,
        stage,
        prim_path: str,
        size: list[int],
        position: list[float],
        rotation: list[float],
        detail: list[int],
    ):
        super().__init__(stage, prim_path, size, position, rotation, detail, 0)


# detail does not affect the flat terrain, the number of vertices is determined by the size
class FlatTerrainBuilder(TerrainBuilder):
    def __init__(
        self,
        base_path: str = None,
        size: list[int] = None,
        position: list[float] = None,
        rotation: list[float] = None,
        detail: list[int] = None
    ):
        if size is None:
            size = [5, 5]
        if position is None:
            position = [0, 0, 0]
        if rotation is None:
            rotation = [0, 0, 0]
        if detail is None:
            detail = [1, 1]

        super().__init__(base_path, size, position, rotation, detail, 0)

    def build(self, stage):
        heightmap = torch.zeros(self.size)

        terrain_path = self._add_heightmap_to_world(heightmap, self.size[0], self.size[1])

        from core.utils.physics import set_physics_properties
        set_physics_properties(terrain_path)

        return FlatTerrainBuild(
            stage,
            terrain_path,
            self.size,
            self.position,
            self.rotation,
            self.detail,
        )
