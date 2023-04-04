##
## Train some agents for open day which I can run inference on and show policy failure.
##

import sys
sys.path.append('src/')

import gym
from stable_baselines3 import PPO

from src import cartpole
# from src.curiosity_ppo import PPO
sys.modules['cartpole'] = cartpole
# from src.icmppo.curiosity.icm import ICM, MlpICMModel
# from src.rewardprediction import MlpRewardModel, RewardPredictor


default_param_dict = {'reset_pole_on_drop': True}
medium_length_param_dict = {'length': 0.65, 'reset_pole_on_drop': True}
medium_masscart_param_dict = {'masscart': 1.3, 'reset_pole_on_drop': True}
medium_length_masscart_param_dict = {'length': 0.65, 'masscart': 1.3, 'reset_pole_on_drop': True}
long_length_param_dict = {'length': 1.0, 'reset_pole_on_drop': True}
heavy_masscart_param_dict = {'masscart': 2.0, 'reset_pole_on_drop': True}
long_heavy_length_masscart_param_dict = {'length': 1.0, 'masscart': 2.0, 'reset_pole_on_drop': True}

param_dicts = [default_param_dict, medium_length_param_dict, medium_masscart_param_dict,
               medium_length_masscart_param_dict, long_length_param_dict,
               heavy_masscart_param_dict, long_heavy_length_masscart_param_dict]

for param_dict in param_dicts:
    # Make env
    env = gym.make("InfHCartPole-v1", param_dict=param_dict)

    # Make agent
    agent = PPO('MlpPolicy', env, verbose=1)

    agent.learn(total_timesteps=300000)
    agent.save(f'agents/params_{"_".join(map(str, param_dict.values()))}.zip')