from pathlib import Path

import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from callbacks.eval_callback import EvalCallback
from common.make_vec_env import make_vec_env
from envs.single_stock_trading_env import StockTradingEnv

TICKER = "SBIN.NS"
TRAIN_FILE = Path("datasets") / f"{TICKER}_train"
EVAL_FILE = Path("datasets") / f"{TICKER}_trade"

CLOSE_PRICES = pl.read_parquet(TRAIN_FILE)["Close"].to_numpy()
EVAL_CLOSE_PRICES = pl.read_parquet(EVAL_FILE)["Close"].to_numpy()


def main():
    model_name = "single_stock_trading_ppo"
    num_envs = 8
    check_env(StockTradingEnv(CLOSE_PRICES, seed=0))

    vec_env = make_vec_env(
        env_id=StockTradingEnv,
        close_prices=CLOSE_PRICES,
        start_seed=1337,
        n_envs=num_envs,
    )

    if Path(f"trained_models/{model_name}.zip").exists():
        reset_num_timesteps = False
        model = PPO.load(
            f"trained_models/{model_name}.zip",
            vec_env,
            print_system_info=True,
            device="auto",
        )
    else:
        reset_num_timesteps = True
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=2,
            # n_steps=128,
            device="auto",
            ent_coef=0.05,
            tensorboard_log="tensorboard_log",
        )

    model.learn(
        total_timesteps=10_000_000,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
        callback=EvalCallback(model_name="ppo"),
        tb_log_name=model_name,
    )
    model.save("trained_models/" + model_name)


if __name__ == "__main__":
    main()
