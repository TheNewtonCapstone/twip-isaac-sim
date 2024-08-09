from perlin_noise import PerlinNoise
from core.terrain.terrain import TerrainBuild, TerrainBuilder
import torch


class PerlinTerrainBuild(TerrainBuild):
    def __init__(
        self,
        container,
        size: list[int],
        position: list[int],
        rotation: list[int],
        detail: list[int],
        height: float,
        bumpiness: float,
    ):
        super().__init__(container, size, position, rotation, detail, height)

        self.bumpiness = bumpiness


class PerlinTerrainBuilder(TerrainBuilder):
    def __init__(
        self,
        size: list[int] = [10, 10],
        position: list[int] = [0, 0, -0.05],
        rotation: list[int] = [0, 0, 0],
        detail: list[int] = [20, 20, 20],
        height: float = 0.1,
        randomize: bool = False,
        bumpiness: float = 24,
    ):
        super().__init__(size, position, rotation, detail, height, randomize)

        self.bumpiness = bumpiness

    def build(self, stage):
        import omni.isaac.core
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr.Usd import Stage

        assert isinstance(stage, Stage), "Container must be of type UsdStage."

        num_rows = int(self.size[0] * self.detail[0])
        num_cols = int(self.size[1] * self.detail[1])

        heightmap = torch.zeros((num_rows, num_cols))

        noise = PerlinNoise(octaves=self.bumpiness)

        for i in range(num_rows):
            for j in range(num_cols):
                heightmap[i, j] = noise([i / num_rows, j / num_cols])

        vertices, triangles = self._heightmap_to_mesh(heightmap, num_cols, num_rows)
        self._add_mesh_to_world(stage, vertices, triangles)

        return PerlinTerrainBuild(
            stage,
            self.size,
            self.position,
            self.rotation,
            self.detail,
            self.height,
            self.bumpiness,
        )
