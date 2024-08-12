import torch
from core.terrain.terrain import TerrainBuild, TerrainBuilder
from core.utils.physics import set_physics_properties
from perlin_noise import PerlinNoise


class PerlinTerrainBuild(TerrainBuild):
    def __init__(
        self,
        stage,
        path: str,
        size: list[int],
        position: list[int],
        rotation: list[int],
        detail: list[int],
        height: float,
        octaves: float,
        noise_scale: float,
    ):
        super().__init__(stage, path, size, position, rotation, detail, height)

        self.octaves = octaves
        self.noise_scale = noise_scale


class PerlinTerrainBuilder(TerrainBuilder):
    def __init__(
        self,
        base_path: str = None,
        size=None,
        position=None,
        rotation=None,
        detail=None,
        height: float = 0.05,
        octaves: float = 12,
        noise_scale: float = 4,
    ):
        if detail is None:
            detail = [40, 40, 40]
        if rotation is None:
            rotation = [0, 0, 0]
        if position is None:
            position = [0, 0, 0]
        if size is None:
            size = [10, 10]

        super().__init__(base_path, size, position, rotation, detail, height)

        self.octaves = octaves
        self.noise_scale = noise_scale

    def build(self, stage):
        num_rows = int(self.size[0] * self.detail[0])
        num_cols = int(self.size[1] * self.detail[1])

        heightmap = torch.zeros((num_rows, num_cols))

        noise = PerlinNoise(octaves=self.octaves)

        for i in range(num_rows):
            for j in range(num_cols):
                heightmap[i, j] = noise([i / num_rows * self.noise_scale, j / num_cols * self.noise_scale])

        terrain_path = self._add_heightmap_to_world(heightmap, num_cols, num_rows)
        set_physics_properties(terrain_path)

        return PerlinTerrainBuild(
            stage,
            terrain_path,
            self.size,
            self.position,
            self.rotation,
            self.detail,
            self.height,
            self.octaves,
            self.noise_scale
        )
