from pathlib import Path

import polars as pl
from sbx import PPO
from stable_baselines3.common.env_checker import check_env

from callbacks.eval_callback import EvalCallback
from common.make_vec_env import make_vec_env
# from envs.single_stock_trading_past_n_price_portfolio_reward_env import StockTradingEnv
# from envs.single_stock_trading_reward_only_at_sell import StockTradingEnv
from envs.single_stock_trading_portfolio_reward_wb import StockTradingEnv

from common.load_close_prices import load_close_prices
from common.set_seed import set_seed

SEED = 1337
set_seed(SEED)

TICKER = "WHIRLPOOL.NS"
TRAIN_FILE = Path("datasets") / f"{TICKER}"

CLOSE_PRICES = load_close_prices(TICKER)

def main():
    model_name = f"portfolio_reward_new_arch_2_wb_{TICKER.split('.')[0]}_jax-ppo"
    print(model_name)
    num_envs = 128
    n_steps = 32
    epoch = 500 // 5
    epoch = 1500
    total_timesteps = (num_envs * n_steps) * epoch
    ent_coef = 0.04
    n_epochs = 20
    check_env(StockTradingEnv(CLOSE_PRICES, seed=SEED))
    vec_env = make_vec_env(
        env_id=StockTradingEnv,
        close_prices=CLOSE_PRICES[:200],
        start_seed=SEED,
        n_envs=num_envs,
    )

    if Path(f"trained_models/{model_name}.zip").exists():
        reset_num_timesteps = False
        model = PPO.load(
            f"trained_models/{model_name}.zip",
            vec_env,
            print_system_info=True,
            device="auto",
            ent_coef=ent_coef,
            n_epochs=n_epochs,
            n_steps=n_steps,
            verbose=0
        )
    else:
        reset_num_timesteps = True
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            n_steps=n_steps,
            device="auto",
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            tensorboard_log="tensorboard_log",
            policy_kwargs = dict(
                net_arch=dict(pi=[1024, 2048, 1024], vf=[1024, 2048, 1024])
            )
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
