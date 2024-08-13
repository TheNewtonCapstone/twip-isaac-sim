from core.terrain.terrain import TerrainBuilder, TerrainBuild


class DefaultGroundPlaneBuild(TerrainBuild):
    def __init__(
        self,
        stage,
        path: str,
    ):
        super().__init__(stage, [], [], 0, [], path)


class DefaultGroundPlaneBuilder(TerrainBuilder):
    @staticmethod
    def build(stage, size=None, resolution=None, height=None, position=None, path="/World/terrains/groundPlane"):
        """
        Notes:
            None of the parameters are used for the default ground plane.
        """

        import omni.isaac.core

        # add a ground plane
        stage.scene.add_default_ground_plane()

        return DefaultGroundPlaneBuild(
            stage,
            path,
        )
