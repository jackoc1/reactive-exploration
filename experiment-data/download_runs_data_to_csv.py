import json

import wandb

api = wandb.Api()
entity, project = "jaoc1", "cartpole_curiosity"

runs = api.runs(f"{entity}/{project}", {
    "state": "finished"
})

for run in runs:
    params = json.loads(run.json_config)
    cartpole_dicts = params["env_params"]["value"]["cartpole_dicts"]

    seed = params["seed"]["value"]
    length, masscart = cartpole_dicts[1]["length"], cartpole_dicts[1]["masscart"]

    history = run.history()
    history.to_csv(f"csv/length_{length}_masscart_{masscart}_seed_{seed}.csv")

