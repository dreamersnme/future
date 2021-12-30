import os

import gym
import argparse
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf
from gym import register

from baselines.sac import sac_learner
from baselines.sac.sac import SoftActorCritic
from baselines.sac.replay_buffer import ReplayBuffer

tf.keras.backend.set_floatx('float64')
logging.basicConfig(level='INFO')

parser = argparse.ArgumentParser(description='SAC')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--env_name', type=str, default='StarTrader-v0', #'MountainCarContinuous-v0',
                    help='name of the gym environment with version')

parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch sample size for training')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to run backprop in an episode')
parser.add_argument('--start_steps', type=int, default=0,
                    help='number of global steps before random exploration ends')
parser.add_argument('--model_path', type=str, default="./sac_log/models",
                    help='path to save model')
parser.add_argument('--model_name', type=str,
                    default=f'{str(datetime.utcnow().date())}-{str(datetime.utcnow().time())}'.replace(':','_'),
                    help='name of the saved model')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--polyak', type=float, default=0.995,
                    help='coefficient for polyak averaging of Q network weights')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')



if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    register(
        id='StarTrader-v0',
        entry_point='gym2.envs.StarTrader:StarTradingEnv',
    )

    args = parser.parse_args()
    #tf.random.set_seed(args.seed)
    # Instantiate the environment.
    env = gym.make(args.env_name, title="SAC", plot_dir="./sac_log/figs")
    env.seed(args.seed)
    sac_learner.learn(env)
