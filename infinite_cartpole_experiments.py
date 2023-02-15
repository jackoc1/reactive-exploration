from pathlib import Path
import os
import sys
sys.path.insert(1, 'src/')

import gym
import torch as th
import wandb
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from src import cartpole
sys.modules['cartpole'] = cartpole

from src.curiosity_ppo import PPO
from src.icmppo.curiosity.icm import ICM, MlpICMModel
from src.rewardprediction import RewardPredictor, MlpRewardModel
from src.icmppo.reporters import TensorBoardReporter


def setup_wandb(dir, name):
    print("Setting up logging to Weights & Biases.")

    # hydra changes working directories
    log_dir = str(Path.joinpath(Path(os.getcwd()), dir))
    # make "wandb" path, otherwise WSL might block writing to dir
    wandb_path = Path.joinpath(Path(log_dir), "wandb")
    wandb_path.mkdir(exist_ok=True, parents=True)
    wandb.login()
    # tracks everything that TensorBoard tracks
    wandb.tensorboard.patch(root_logdir=log_dir)
    wandb_run = wandb.init(project="curiosity_proportions_inf_cartpole", name=name,
                           dir=log_dir, save_code=False, config={'override_dirname': True})
    # wandb_run.tags = wandb_run.tags + tuple([seed])
    print(f"Writing Weights & Biases logs to: {str(wandb_path)}")
    return wandb_run


def make_callbacks(save_dir=None, save_freq=None):
    print('Making callbacks...')
    callbacks = []
    if save_dir and save_freq:
        callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=save_dir, name_prefix='', verbose=1))
    return CallbackList(callbacks)


def make_agent(env, seed, dir):
    print("Making agent...")
    exploration_reporter = TensorBoardReporter(logdir=dir + '/' + str(seed)) if dir else None
    exp_model = MlpICMModel

    exploration_factory = ICM.factory(exp_model.factory(), reporter=exploration_reporter, 
                                      intrinsic_reward_integration=0.85, policy_weight=1.0,
                                      reward_scale=0.01, weight=0.2)

    # TODO: Reward model might not be necessary since I don't think reward function changes.
    reward_model_reporter = TensorBoardReporter(logdir=dir + '/RewardModel') if dir else None
    reward_model = MlpRewardModel
    reward_predictor_factory = RewardPredictor.factory(reward_model.factory(), reporter=reward_model_reporter,
                                                       intrinsic_reward_integration=0.15, intrinsic_reward_scale=1.0,
                                                       norm_rewards= False)

    agent = PPO(policy="MlpPolicy", env=env, policy_kwargs=None, verbose=1,
                        tensorboard_log=dir, seed=seed, curiosity_factory=exploration_factory,
                        reward_predictor_factory=reward_predictor_factory, ent_coef=0)

    return agent

def main(run_name, env_name, dir='infh-cartpole', cartpole_dicts={}, use_wandb=True, total_timesteps=1e6, step_modulo=None, save_freq=2.5e6, save=True):
    if use_wandb:
        setup_wandb(dir, name=run_name)
    
    env = gym.make(env_name, param_dict=cartpole_dicts, change_interval=step_modulo)
    agent = make_agent(env, seed=None, dir=dir)
    callbacks = make_callbacks(dir + '/callbacks', save_freq=save_freq)

    print('Starting training...')
    agent.learn(total_timesteps=total_timesteps, callback=callbacks, log_interval=1)

    if save:
        agent.save(Path.joinpath(Path(os.getcwd()), f'/saved_models/{run_name}'))

    if use_wandb:
        # necessary for Hydra multiruns
        wandb.finish()
        wandb.tensorboard.unpatch()

if __name__ == '__main__':
    cartpole_dicts = {'modulo_theta': True}
    main('BaseCurioisityPPO_BaseInfHCartPole', 'InfHCartPole-v1', cartpole_dicts=cartpole_dicts, use_wandb=True)
    