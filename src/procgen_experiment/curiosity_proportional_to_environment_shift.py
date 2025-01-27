from pathlib import Path
import os
import sys
sys.path.insert(1, 'src/')

import gym
import wandb
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from src.custom_curiosity_ppo import CustomPPO as PPO
from src.icmppo.curiosity.icm import ICM, CNNICMModel, MlpICMModel
from src.rewardprediction import RewardPredictor, RewardModel, MlpRewardModel
from src.icmppo.reporters import TensorBoardReporter
from src.custom_models import VisionCNN


def setup_wandb(seed, dir):
    print("Setting up logging to Weights & Biases.")

    # hydra changes working directories
    log_dir = str(Path.joinpath(Path(os.getcwd()), dir))
    # make "wandb" path, otherwise WSL might block writing to dir
    wandb_path = Path.joinpath(Path(log_dir), "wandb")
    wandb_path.mkdir(exist_ok=True, parents=True)
    wandb.login()
    # tracks everything that TensorBoard tracks
    wandb.tensorboard.patch(root_logdir=log_dir)
    wandb_run = wandb.init(project="curiosity_proportions", name=log_dir,
                           dir=log_dir, save_code=False)
    wandb_run.tags = wandb_run.tags + (seed)
    print(f"Writing Weights & Biases logs to: {str(wandb_path)}")
    return wandb_run

def make_agent(env, seed, dir):
    print("Making agent...")
    is_cnn_policy = True
    # if is_cnn_policy:
    #     policy_kwargs = dict(
    #         features_extractor_class=VisionCNN,
    #         features_extractor_kwargs=dict(features_dim=128),
    #     )
    # else:
    #     policy_kwargs = None
    
    exploration_reporter = TensorBoardReporter(logdir=dir + '/' + str(seed)) if dir else None
    exp_model = CNNICMModel if is_cnn_policy else MlpICMModel

    exploration_factory = ICM.factory(exp_model.factory(), reporter=exploration_reporter, 
                                      intrinsic_reward_integration=0.85, policy_weight=1.0,
                                      reward_scale=0.01, weight=0.2)

    reward_model_reporter = TensorBoardReporter(logdir=dir + '/RewardModel') if dir else None
    reward_model = RewardModel if is_cnn_policy else MlpRewardModel
    reward_predictor_factory = RewardPredictor.factory(reward_model.factory(), reporter=reward_model_reporter,
                                                       intrinsic_reward_integration=0.15, intrinsic_reward_scale=1.0,
                                                       norm_rewards= False)

    agent = PPO(policy="CnnPolicy", env=env, policy_kwargs=None, verbose=1,
                        tensorboard_log=dir, seed=seed, curiosity_factory=exploration_factory,
                        reward_predictor_factory=reward_predictor_factory, ent_coef=0)


def make_callbacks(save_dir=None, save_freq=None):
    print('Making callbacks...')
    callbacks = []
    if save_dir and save_freq:
        callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=save_dir, name_prefix='', verbose=1))
    return CallbackList(callbacks)


def main(use_wandb=True, seed=0, dir="unique_dir_name"):
    if use_wandb:
        setup_wandb(seed, dir)

    env = gym.make('procgen:procgen-coinrun-v0', seed=seed)
    agent = make_agent(env, seed, dir)
    callbacks = make_callbacks(dir + '/callbacks', save_freq=2.5e7)
    print('Starting training...')
    agent.learn(total_timesteps=2e6, callback=callbacks, log_interval=1)

    if use_wandb:
        # necessary for Hydra multiruns
        wandb.finish()
        wandb.tensorboard.unpatch()

if __name__ == "__main__":
    main()
