from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
from .multiagentenv import MultiAgentEnv
from .test_env import PursuitEnv
from .unrealcv_env import UnrealCVEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["test_env"] = partial(env_fn, env=PursuitEnv)
REGISTRY["unrealcv_env"] = partial(env_fn, env=UnrealCVEnv)

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
