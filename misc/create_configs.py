# Procedurally generate envrionment parameteres where the first 500k steps of the environment are
# default cartpole parameters and then next 500k steps length and masscart are incremented by
# a fixed amount. 

import yaml

with open('1.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

i = 169
for l in range(1,5):
    config["cartpole_dicts"][1]["length"] = 0.5 # reset

    config["cartpole_dicts"][1]["length"] = round(config["cartpole_dicts"][1]["length"] - 0.1*l, 1)
    for mc in range(1,5):
        config["cartpole_dicts"][1]["masscart"] = 1.0 # reset
        config["cartpole_dicts"][1]["masscart"] = round(config["cartpole_dicts"][1]["masscart"] - 0.2*mc, 1)
        i += 1
        with open(f"{i}.yaml", "w") as file:
            yaml.dump(config, file)
