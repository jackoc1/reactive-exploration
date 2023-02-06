import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('procgen:procgen-coinrun-v0', distribution_mode="easy", start_level=0, num_levels=1, render=True)
model = PPO.load('procgen/callbacks/_10000000_steps.zip')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
