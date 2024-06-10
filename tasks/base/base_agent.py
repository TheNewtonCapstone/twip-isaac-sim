import abc


from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics


class BaseAgent(object):
    def __init__(self, _config) -> None:
        self.config = _config
        pass

    def construct(self, stage) -> bool:
        import omni.kit.commands

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

        status, stage_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="/home/nyquist/projects/twip-isaac-sim/tasks/twip/assets/twip.urdf",
            import_config=import_config,
        )

        # lwd =
