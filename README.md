# Intinsic Quantification of Domain-Shift Magnitude in Reinforcement Learning

__Work submitted towards the completion of the degree of Bachelor of Science in Data Science and Analytics, April 2023.__

Jack O'Connor, 119319446, Department of Computer Science, University College Cork.

## Why base my work on Reactive Exploration?

The authors of Reactive Exploration showed the potential for the Reactive Exploration algorithm to detect domain shifts.

The hypothesis under investigation in my work is that the reward loss metric tracked by Reactive Exploration can not only detect domain shifts, 
but is also capable of quantifying them i.e. the maginitude of the reward loss at time of domain-shift can directly predict future policy performance 
in the domain shifted environment.


## Additions to existing repo

### configs/env_params/domain_shift_quantification/ 

Procedurally generated cartpole environment specifications used to track how Reactive Exploration metrics change as the domain shift magnitude increases.

### experiment-data/

Script to download data from wandb in csv format; jupyter notebook to read these csvs into a multidimensional array object and functions to extract insights from this object, as well as plots analysing this data; and the full experiment data multidimensional array in pickled form.

### src/custom_curiosity_ppo.py

Modified PPO policy equipped with ICM which can dynamically alter its hyperparameters based on online training metrics 
from Reactive Exploration algorithm. (Not used due to time constraints)

### misc/

Miscellaneous scripts for scheduling experiments, creating open day demos, generating yaml configs, etc.
