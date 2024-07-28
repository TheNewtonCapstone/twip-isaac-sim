import yaml

config = {}
with open("configs/randomization.yaml", "r") as f:
    config = yaml.safe_load(f)

domain_randomization = config["randomization_params"]["twip"][
    "articulation_view_properties"
]

args = {}

for prop in domain_randomization:

    from_x = []
    to_y = []
    prop_range = domain_randomization[prop]["range"]
    if isinstance(prop_range[0], list):
        for item in prop_range:
            from_x.append(item[0])
            to_y.append(item[1])
    else:
        from_x = [prop_range[0]]
        to_y = [prop_range[1]]

    args[prop] = (tuple(from_x), tuple(to_y))

print(args)

# rep.distribution.normal((0.0, 0.0, 0.0), (0.2, 0.2, 0.0))
