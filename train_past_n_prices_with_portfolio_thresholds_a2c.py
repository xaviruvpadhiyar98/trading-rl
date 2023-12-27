from pathlib import Path

import polars as pl
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

from callbacks.eval_callback import EvalCallback
from common.make_vec_env import make_vec_env
# from envs.single_stock_trading_past_n_price_portfolio_reward_env import StockTradingEnv
from envs.single_stock_trading_reward_only_at_sell import StockTradingEnv

from common.load_close_prices import load_close_prices
from common.set_seed import set_seed

SEED = 1337
set_seed(SEED)

TICKER = "WHIRLPOOL.NS"
TRAIN_FILE = Path("datasets") / f"{TICKER}"

CLOSE_PRICES = load_close_prices(TICKER)


def main():
    model_name = f"portfolio_reward_{TICKER.split('.')[0]}_a2c"
    num_envs = 128
    n_steps = 5
    epoch = 10000 * 4
    total_timesteps = (num_envs * n_steps) * epoch

    check_env(StockTradingEnv(CLOSE_PRICES, seed=0))
    vec_env = make_vec_env(
        env_id=StockTradingEnv,
        close_prices=CLOSE_PRICES[:200],
        start_seed=SEED,
        n_envs=num_envs,
    )

    if Path(f"trained_models/{model_name}.zip").exists():
        reset_num_timesteps = False
        model = A2C.load(
            f"trained_models/{model_name}.zip",
            vec_env,
            print_system_info=True,
            device="cpu",
        )
    else:
        reset_num_timesteps = True
        model = A2C(
            "MlpPolicy",
            vec_env,
            verbose=2,
            n_steps=n_steps,
            device="cpu",
            ent_coef=0.05,
            tensorboard_log="tensorboard_log",
        )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
        callback=EvalCallback(model_name=model_name),
        tb_log_name=model_name,
    )
    model.save("trained_models/" + model_name)


if __name__ == "__main__":
    main()
