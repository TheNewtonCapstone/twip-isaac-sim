from core.terrain.terrain import TerrainBuilder, TerrainBuild


class DefaultGroundPlaneBuild(TerrainBuild):
    def __init__(
        self,
        container,
        size: list[int],
        position: list[float],
        rotation: list[float],
        detail: list[int],
        height: float,
    ):
        super().__init__(container, size, position, rotation, detail, height)


class DefaultGroundPlaneBuilder(TerrainBuilder):
    def build(self, stage):
        import omni.isaac.core

        # add a ground plane
        stage.scene.add_default_ground_plane()

        return DefaultGroundPlaneBuild(
            stage,
            self.size,
            self.position,
            self.rotation,
            self.detail,
            self.height,
        )
