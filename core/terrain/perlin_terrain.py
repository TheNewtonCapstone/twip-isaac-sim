import torch
from core.terrain.terrain import TerrainBuild, TerrainBuilder
from perlin_noise import PerlinNoise


class PerlinTerrainBuild(TerrainBuild):
    def __init__(
        self,
        stage,
        size: list[float],
        resolution: list[int],
        height: float,
        position: list[float],
        path: str,
        octaves: float,
        noise_scale: float,
    ):
        super().__init__(stage, size, resolution, height, position, path)

        self.octaves = octaves
        self.noise_scale = noise_scale


class PerlinTerrainBuilder(TerrainBuilder):
    @staticmethod
    def build(
        stage,
        size=None,
        resolution=None,
        height=0.05,
        position=None,
        path="/World/terrains",
        octaves=12,
        noise_scale=4
    ):
        if size is None:
            size = [20, 20]
        if resolution is None:
            resolution = [40, 40]
        if position is None:
            position = [0, 0, 0]

        num_rows = int(size[0] * resolution[0])
        num_cols = int(size[1] * resolution[1])

        heightmap = torch.zeros((num_rows, num_cols))

        noise = PerlinNoise(octaves=octaves)

        for i in range(num_rows):
            for j in range(num_cols):
                heightmap[i, j] = noise([i / num_rows * noise_scale, j / num_cols * noise_scale])

        terrain_path = TerrainBuilder._add_heightmap_to_world(
            heightmap,
            size,
            num_cols,
            num_rows,
            height,
            path,
            "perlin",
            position
        )

        from core.utils.physics import set_physics_properties
        set_physics_properties(terrain_path)

        return PerlinTerrainBuild(
            stage,
            size,
            resolution,
            height,
            position,
            terrain_path,
            octaves,
            noise_scale
        )
