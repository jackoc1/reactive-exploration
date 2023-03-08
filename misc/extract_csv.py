import yaml
from pathlib import Path



test_dir = "/fyp/outputs/PPO_Cartpole_Mass-v1/2023-02-14_10-04-54/double_mass/wandb/run-20230214_100455-1s8cx16c/files"
config_name = "config.yaml"
filepath = Path.joinpath(test_dir, config_name)

with open(filepath, 'r') as stream:
    config = yaml.safe_load(stream)


cartpole_dicts = d["env_params"]["value"]["cartpole_dicts"]
if len(cartpole_dicts) == 1:
    pass
elif len(cartpole_dicts) == 2:
    pass
else:
    raise("More than 2 cartpole dicts")

