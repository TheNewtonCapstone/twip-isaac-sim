import yaml

def format_randomization_params(twip_params):
    
    def process_property(distribution, range_values):
        range_str = get_randomization_range(range_values)
        if distribution == "uniform":
            return f"rep.distribution.uniform({range_str})"
        elif distribution == "normal":
            return f"rep.distribution.normal({range_str})"
        else:
            raise ValueError(f"Invalid distribution type: {distribution}")

    def format_properties(properties):
        return {
            prop: process_property(
                prop_data.get('distribution', 'uniform'),
                prop_data.get('range', [])
            ) for prop, prop_data in properties.items()
        }

    formatted_params = {}

    # Extract relevant sections from twip_params
    for gate_type in twip_params:
        gate_type_config = twip_params.get(gate_type, {})
        for property_type in ["articulation_view_properties", "dof_properties", "rigid_body_properties"]:
            property_config = gate_type_config.get(property_type, {})
            
            # Format properties for each type
            formatted_params[gate_type] = {}
            formatted_params[gate_type][property_type] = {
                'additive': format_properties(property_config.get('additive', {})),
                'scaling': format_properties(property_config.get('scaling', {})),
                'direct': format_properties(property_config.get('direct', {}))
            }

    return formatted_params

# Example usage:
# formatted_params = format_randomization_params(twip_params, rep)


def get_randomization_range(prop_range):
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

def main():
    config = {}
    with open("configs/randomization.yaml", "r") as f:
        config = yaml.safe_load(f)

    domain_randomization = config["randomization_params"]

    twip_params = domain_randomization.get("twip", {})
    print(twip_params)

    print("Formated data: ", format_randomization_params(twip_params))


if __name__ == "__main__":
    main()
