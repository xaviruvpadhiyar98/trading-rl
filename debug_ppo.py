from pathlib import Path

import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from callbacks.eval_callback import EvalCallback
from common.make_vec_env import make_vec_env
from envs.single_stock_trading_past_n_price_portfolio_reward_env import StockTradingEnv
import numpy as np
from common.load_close_prices import load_close_prices
from common.set_seed import set_seed

SEED = 1337
set_seed(SEED)

TICKER = "WHIRLPOOL.NS"
TRAIN_FILE = Path("datasets") / f"{TICKER}"

CLOSE_PRICES = load_close_prices(TICKER)

def main():
    model_name = f"single_stock_trading_portfolio_reward_{TICKER.split('.')[0]}_ppo"
    num_envs = 256
    n_steps = 128
    epoch = 10
    total_timesteps = (num_envs * n_steps) * epoch
    check_env(StockTradingEnv(CLOSE_PRICES, seed=0))
    vec_env = make_vec_env(
        env_id=StockTradingEnv,
        close_prices=CLOSE_PRICES,
        start_seed=SEED,
        n_envs=num_envs,
    )

    reset_num_timesteps = True
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=2,
        device="auto",
        ent_coef=0.05,
        n_steps=n_steps
    )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
        callback=EvalCallback(model_name=model_name),
        tb_log_name=model_name,
    )

    env = StockTradingEnv(CLOSE_PRICES, SEED)
    obs, info = env.reset()
    states = None
    deterministic = False
    episode_starts = np.ones((1,), dtype=bool)
    infos = []
    while True:
        actions, states = model.predict(
            obs,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        obs, reward, done, truncated, info = env.step(actions.item())
        infos.append(info)
        if done or truncated:
            print(info['counter'])
            break



if __name__ == "__main__":
    main()
