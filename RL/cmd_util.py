"""
Helpers for scripts like run_atari.py.
"""



import gym
from gym.wrappers import FlattenObservation
from baselines.bench import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.wrappers import ClipActionsWrapper

def make_vec_env(env_id,
                 env_kwargs=None,
                 flatten_dict_observations=True):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    env_kwargs = env_kwargs or {}
    def make_thunk():
        return lambda: make_env(
            env_id=env_id,
            flatten_dict_observations=flatten_dict_observations,
            env_kwargs=env_kwargs,
        )

    return DummyVecEnv([make_thunk()])


def make_env(env_id,flatten_dict_observations=True,  env_kwargs=None):
    env_kwargs = env_kwargs or {}

    env = gym.make(env_id, **env_kwargs)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = Monitor(env,None,
                  allow_early_resets=True)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    return env
