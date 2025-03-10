from .multiagentenv import MultiAgentEnv
import os

import gym
import gym_unrealcv


class UnrealCVEnv(MultiAgentEnv):
    def __new__(cls, env_id, **kwargs):
        env = gym.make(env_id, **kwargs)
        print('Build Env ', env_id)
        return env
