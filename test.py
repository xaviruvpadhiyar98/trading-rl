import json
from pathlib import Path

import polars as pl
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from common.make_vec_env import make_vec_env
from envs.single_stock_trading_past_n_price_env import StockTradingEnv
import numpy as np


TICKER = "SBIN.NS"
EVAL_FILE = Path("datasets") / f"{TICKER}_trade"
df = pl.read_parquet(EVAL_FILE)
EVAL_CLOSE_PRICES = (
    df.with_columns(index=pl.int_range(0, end=df.shape[0], eager=True))
    .sort("index")
    .set_sorted("index")
    .group_by_dynamic(
        "index", every="1i", period="40i", include_boundaries=True, closed="right"
    )
    .agg(pl.col("Close"))
    .with_columns(pl.col("Close").list.len().alias("Count"))
    .filter(pl.col("Count") == 40)["Close"]
    .to_numpy()
)

eval_vec_env = make_vec_env(
    env_id=StockTradingEnv,
    close_prices=EVAL_CLOSE_PRICES,
    start_seed=1337,
    n_envs=1,
)

env = StockTradingEnv(EVAL_CLOSE_PRICES, 1337)
model_name = "single_stock_trading_past_n_prices_a2c"
trade_model = A2C.load(
    "trained_models/" + model_name, eval_vec_env, print_system_info=True
)
obs = eval_vec_env.reset()
states = None
deterministic = True
episode_starts = np.ones((1,), dtype=bool)
while True:
    actions, states = trade_model.predict(
        obs,
        state=states,
        episode_start=episode_starts,
        deterministic=deterministic,
    )
    obs, rewards, dones, infos = eval_vec_env.step(actions)
    episode_starts[0] = dones[0]
    print(infos[0]["portfolio_value"], infos[0]["description"])
    if dones[0]:
        break
