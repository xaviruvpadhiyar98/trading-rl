from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def make_vec_env(env_id, close_prices, start_seed, n_envs):
    def make_env(rank: int):
        def _init():
            env = env_id(close_prices, start_seed + rank)
            env = _patch_env(env)
            env.action_space.seed(start_seed + rank)
            env = Monitor(env)
            return env

        return _init

    start_index = 0
    vec_env = DummyVecEnv([make_env(i + start_index) for i in range(n_envs)])
    return vec_env
