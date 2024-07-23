from core.terrain.terrain import TerrainBuilder, TerrainBuild


class DefaultGroundPlaneBuild(TerrainBuild):
    def __init__(
        self,
        container,
        size: list[int],
        position: list[int],
        rotation: list[int],
        scale: list[int],
    ):
        super().__init__(container, size, position, rotation, scale)


class DefaultGroundPlaneBuilder(TerrainBuilder):
    def __init__(self):
        super().__init__()

    def build(self, container):
        import omni.isaac.core
        from omni.isaac.core import World

        assert isinstance(container, World), "Container must be of type World."

        # add a ground plane
        container.scene.add_default_ground_plane()

        return DefaultGroundPlaneBuild(
            container,
            self.position,
            self.rotation,
            self.scale,
        )
