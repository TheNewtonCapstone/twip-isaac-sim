import abc

from tasks.base.base_agent import BaseAgent


class TwipAgent(BaseAgent):
    def __init__(self, _config) -> None:
        super().__init__(_config)

    def construct(self, stage) -> bool:
        super().construct(stage)

        # these only work after SimulationApp is initialized (to be done in scripts that import this class)
        from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics
        import omni.kit.commands

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = False
        import_config.fix_base = False

        status, stage_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="/home/nyquist/projects/twip-isaac-sim/tasks/twip/assets/twip.urdf",
            import_config=import_config,
            get_articulation_root=True,
        )
        self.stage_path = stage_path

        lwd = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(self.stage_path + "/lwheel"), "angular"
        )
        rwd = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(self.stage_path + "/rwheel"), "angular"
        )

        lwd.GetTargetVelocityAttr().Set(50)
        rwd.GetTargetVelocityAttr().Set(50)

        lwd.GetDampingAttr().Set(15000)
        rwd.GetDampingAttr().Set(15000)

        lwd.GetStiffnessAttr().Set(0)
        rwd.GetStiffnessAttr().Set(0)

    def pre_physics(self, _sim_app) -> None:
        # this only works after SimulationApp is initialized (to be done in scripts that import this class)
        from omni.isaac.core.articulations import Articulation

        super().pre_physics(_sim_app)

        art = Articulation(prim_path=self.stage_path)
        art.initialize()
