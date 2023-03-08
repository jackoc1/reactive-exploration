import yaml

with open('0.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

i = 0
for l in range(8):
    config["cartpole_dicts"][1]["length"] = 1.0 # reset

    config["cartpole_dicts"][1]["length"] = round(config["cartpole_dicts"][1]["length"] + 0.1*l, 1)
    for mc in range(8):
        config["cartpole_dicts"][1]["masscart"] = 1.0 # reset
        config["cartpole_dicts"][1]["masscart"] = round(config["cartpole_dicts"][1]["masscart"] + 0.2*mc, 1)
        i += 1
        with open(f"{i}.yaml", "w") as file:
            yaml.dump(config, file)
