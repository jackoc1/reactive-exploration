##
## Run inference and switch between envs to showcase agent policy failure.
## This is all purely for illustrative purposes. Numbers are selected for
## getting a certain type of video output.
##

import sys
sys.path.append('src/')

import gym
from stable_baselines3 import PPO

from src import cartpole
sys.modules['cartpole'] = cartpole



default_param_dict = {'length': 1.5}
medium_length_param_dict = {'length': 3.0}
medium_masscart_param_dict = {'length': 1.5, 'masscart': 5.0}
medium_length_masscart_param_dict = {'length': 3.0, 'masscart': 5.0}
long_length_param_dict = {'length': 4.0}
heavy_masscart_param_dict = {"length": 1.5, 'masscart': 8.0}
long_heavy_length_masscart_param_dict = {'length': 4.0, 'masscart': 8.0}
light_masscart_param_dict = {'length': 1.5, 'masscart': 0.2}
medium_light_length_masscart_param_dict = {'length': 3.0, 'masscart': 0.2}
long_light_length_masscart_param_dict = {'length': 4.0, 'masscart': 0.2}

alternate_param_dicts = [default_param_dict,
                         medium_length_param_dict,
                         medium_masscart_param_dict,
                         medium_length_masscart_param_dict,
                         long_length_param_dict,
                         heavy_masscart_param_dict,
                         long_heavy_length_masscart_param_dict,
                         light_masscart_param_dict,
                         medium_light_length_masscart_param_dict,
                         long_light_length_masscart_param_dict]

default_agent = PPO.load('agents/params_True.zip')

agent = default_agent
env = gym.make("InfHCartPole-v1", param_dict=default_param_dict)
obs = env.reset()
actions=[]

time = 500
num_alternates = len(alternate_param_dicts)
i = 0
while True:
    env.render()
    action = agent.predict(obs)[0]
    obs, reward, done, info = env.step(action)
    actions.append(action)

    if i % time == 0 and i != 0:
        env.update_environment_parameters(alternate_param_dicts[(i // time) % num_alternates])
        env.reset()
    i += 1
