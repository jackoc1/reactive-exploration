# Use python to schedule run_experiment.sh to run after the previous run_experiment.sh script has terimated.
# Iterate through all procedurally generated env params.

import os

for i in range(170, 186):
  code = os.system("bash -l /home/jaoc1/fyp/reactive-exploration/misc/run_experiment.sh -c {}".format(i))
  if code != 0:
    with open("unfinished_configs.txt", 'a') as file:
      file.write("Config {} did not run\n".format(i))

