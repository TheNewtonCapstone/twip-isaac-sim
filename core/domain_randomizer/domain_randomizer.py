import torch
import numpy as np


class DomainRandomizer:
    def __init__(self, world, twip_art_view, randomization_params):

        import omni.replicator.isaac as dr
        import omni.replicator.core as rep

        self.dr = dr
        self.rep = rep

        self.world = world
        self.twip_art_view = twip_art_view
        self.randomization_params = randomization_params
        self.frequency = randomization_params.get("frequency", 1)
        self.domain_params = randomization_params.get("twip", {})

        self.num_dof = self.twip_art_view.num_dof
        self.rigid_body_names = self.twip_art_view.body_names
        print("rigid_body_names: ", self.rigid_body_names)

        self.on_interval_properties = {}
        self.on_reset_properties = {}

        self.format_randomization_params()

        self.frame_idx = 0

        # Register the simulation context and articulation view
        self.dr.physics_view.register_simulation_context(self.world)
        self.dr.physics_view.register_articulation_view(self.twip_art_view)
        print("Registered simulation context and articulation view")

    def format_randomization_params(self):
        def process_property(distribution, range_values):
            range_str = self.get_randomization_range(range_values)
            if distribution == "uniform":
                return self.rep.distribution.uniform(
                    tuple(range_str[0]), tuple(range_str[1])
                )
            elif distribution == "normal":
                return self.rep.distribution.normal(
                    tuple(range_str[0]), tuple(range_str[1])
                )
            else:
                raise ValueError(f"Invalid distribution type: {distribution}")

        def format_properties(properties):
            return {
                prop: process_property(
                    prop_data.get("distribution", "uniform"), prop_data.get("range", [])
                )
                for prop, prop_data in properties.items()
            }

        formatted_params = {}

        # Extract relevant sections from twip_params
        for gate_type in self.domain_params:
            gate_type_config = self.domain_params.get(gate_type, {})
            formatted_params[gate_type] = {}

            for property_type in [
                "articulation_view_properties",
                "dof_properties",
                "rigid_body_properties",
            ]:
                property_config = gate_type_config.get(property_type, {})

                formatted_params[gate_type][property_type] = {
                    "additive": format_properties(property_config.get("additive", {})),
                    "scaling": format_properties(property_config.get("scaling", {})),
                    "direct": format_properties(property_config.get("direct", {})),
                }

        self.on_interval_properties = formatted_params.get("on_interval", {})
        self.on_reset_properties = formatted_params.get("on_reset", {})

    def get_randomization_range(self, prop_range):
        from_x = []
        to_y = []
        if isinstance(prop_range[0], list):
            for item in prop_range:
                from_x.append(item[0])
                to_y.append(item[1])
        else:
            from_x = [prop_range[0]]
            to_y = [prop_range[1]]

        return from_x, to_y

    def apply_randomization(self):
        with self.dr.trigger.on_rl_frame(num_envs=self.twip_art_view.count):
            with self.dr.gate.on_interval(interval=self.frequency):
                for body in self.on_interval_properties:
                    if "articulation_view_properties" in body:
                        for prop in self.on_interval_properties[body]:
                            body_properties = self.on_interval_properties.get(body, {})
                            args = body_properties.get(prop, {})
                            self.dr.physics_view.randomize_articulation_view(
                                view_name=self.twip_art_view.name,
                                operation=str(prop),
                                **args,
                            )

            with self.dr.gate.on_env_reset():
                for body in self.on_reset_properties:
                    if "articulation_view_properties" in body:
                        for prop in self.on_reset_properties[body]:
                            body_properties = self.on_reset_properties.get(body, {})
                            args = body_properties.get(prop, {})
                            self.dr.physics_view.randomize_articulation_view(
                                view_name=self.twip_art_view.name,
                                operation=str(prop),
                                **args,
                            )

    def step_randomization(self):
        reset_inds = []
        if self.frame_idx % 200 == 0:
            reset_inds = np.arange(self.twip_art_view.count)
        self.dr.physics_view.step_randomization(reset_inds=reset_inds)
        self.frame_idx += 1
