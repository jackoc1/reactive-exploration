import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('procgen:procgen-coinrun-v0', distribution_mode="easy", start_level=0, num_levels=1, render=True)
# model = PPO.load('procgen/callbacks/sb3_PPO/_10000000_steps.zip')
model = DQN.load('procgen/callbacks/sb3_DQN/_10000000_steps.zip')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)
