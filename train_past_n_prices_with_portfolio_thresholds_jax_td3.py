from pathlib import Path

from sbx import TD3
from stable_baselines3.common.env_checker import check_env

from callbacks.eval_callback import EvalCallback
from common.make_vec_env import make_vec_env
# from envs.single_stock_trading_past_n_price_portfolio_reward_env import StockTradingEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from envs.single_stock_trading_reward_only_at_sell import StockTradingEnv

from common.load_close_prices import load_close_prices
from common.set_seed import set_seed
import numpy as np

SEED = 1337
set_seed(SEED)

TICKER = "WHIRLPOOL.NS"
TRAIN_FILE = Path("datasets") / f"{TICKER}"

CLOSE_PRICES = load_close_prices(TICKER)

def main():
    model_name = f"single_stock_trading_portfolio_reward_{TICKER.split('.')[0]}_jax-td3"
    num_envs = 1024 * 2
    n_steps = 128
    epoch = 500 // 5
    total_timesteps = (num_envs * n_steps) * epoch
    check_env(StockTradingEnv(CLOSE_PRICES, seed=SEED))
    vec_env = make_vec_env(
        env_id=StockTradingEnv,
        close_prices=CLOSE_PRICES[:200],
        start_seed=SEED,
        n_envs=num_envs,
    )
    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    if Path(f"trained_models/{model_name}.zip").exists():
        reset_num_timesteps = False
        model = TD3.load(
            f"trained_models/{model_name}.zip",
            vec_env,
            print_system_info=True,
            device="auto",
        )
        # model.ent_coef = 0.0 # now let's exploit
        # model.use_sde = True
    else:
        reset_num_timesteps = True
        model = TD3(
            "MlpPolicy",
            vec_env,
            # learning_rate=0.00001,
            verbose=2,
            action_noise=action_noise,
            # n_steps=n_steps,
            # device="auto",
            # ent_coef=0.15,
            # use_sde=True,
            # sde_sample_freq=4,
            # vf_coef=0.5,
            # max_grad_norm=0.5,
            # normalize_advantage=True,
            # n_epochs=20,
            tensorboard_log="tensorboard_log",
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[512, 512], qf=[512, 512]
                ),
                # log_std_init=2
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
