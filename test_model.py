from pathlib import Path

import polars as pl
from sbx import PPO as SBXPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from callbacks.eval_callback import EvalCallback
from common.make_vec_env import make_vec_env
# from envs.single_stock_trading_past_n_price_portfolio_reward_env import StockTradingEnv
# from envs.single_stock_trading_reward_only_at_sell import StockTradingEnv
from envs.single_stock_trading_portfolio_reward_wb import StockTradingEnv

from common.load_close_prices import load_close_prices
from common.set_seed import set_seed
import numpy as np

SEED = 1337
set_seed(SEED)

TICKER = "WHIRLPOOL.NS"
TRAIN_FILE = Path("datasets") / f"{TICKER}"

CLOSE_PRICES = load_close_prices(TICKER)

def main():
    model_name = f"portfolio_reward_new_arch_2_wb_{TICKER.split('.')[0]}_jax-ppo"
    print(model_name)
    check_env(StockTradingEnv(CLOSE_PRICES, seed=SEED))
    eval_vec_env = make_vec_env(
        env_id=StockTradingEnv,
        close_prices=CLOSE_PRICES[:200],
        start_seed=SEED,
        n_envs=1,
    )

    trade_model = PPO.load(
        f"trained_models/{model_name}.zip",
        eval_vec_env,
        print_system_info=True,
        device="auto",
        verbose=0
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
        # print(infos[0])
        print(infos[0]["portfolio_value"], infos[0]["description"])
        print()
        if dones[0]:
            break



if __name__ == "__main__":
    main()
