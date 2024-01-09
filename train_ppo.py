from pathlib import Path

from sbx import PPO as SBXPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize

from common.make_vec_env import make_vec_env
from envs.single_stock_trading_portfolio_reward_wb import StockTradingEnv

from common.load_close_prices import load_close_prices
# from common.set_seed import set_seed
import numpy as np
import json





SEED = 1337
# set_seed(SEED)

TICKER = "WHIRLPOOL.NS"
TRAIN_FILE = Path("datasets") / f"{TICKER}"

CLOSE_PRICES = load_close_prices(TICKER)
NUM_ENVS = 512
N_STEPS = 32
EPOCH = 500 // 5
EPOCH = 100
TOTAL_TIMESTEPS = (NUM_ENVS * N_STEPS) * EPOCH
ENT_COEF = 0.07
N_EPOCHS = 50



class EvalCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def test_and_log(self, seed) -> None:

        env = StockTradingEnv(CLOSE_PRICES, seed)
        trade_model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            n_steps=N_STEPS,
            batch_size=N_STEPS,
            device="auto",
            n_epochs=N_EPOCHS,
            ent_coef=ENT_COEF,
            policy_kwargs = dict(
                net_arch=dict(pi=[1024, 2048, 1024], vf=[1024, 2048, 1024])
            )
        )
        trade_model.set_parameters(self.model.get_parameters())
        obs, info = env.reset()
        states = None
        deterministic = False
        episode_starts = np.ones((1,), dtype=bool)
        infos = []
        while True:
            actions, states = trade_model.predict(
                obs,
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            obs, reward, done, truncated, info = env.step(actions.item())
            # infos.append(info)
            if done or truncated:
                # if info['portfolio_value'] >= 12000:
                #     dir = Path(f"logs/{self.log_name}/")
                #     dir.mkdir(parents=True, exist_ok=True)
                #     (dir / f"{self.num_timesteps}-{info['portfolio_value']}").write_text(
                #         json.dumps(infos, default=str, indent=4)
                #     )
                print(info["seed"], info['counter'], info['portfolio_value'], info['reward_tracker'], info['description'].replace("<br>", "\n"), info["starting_random_number"])
                self.logger.record(f"trade/ep_len", info['counter'])
                self.logger.record(f"trade/portfolio_value", info['portfolio_value'])
                self.logger.record(f"trade/reward_tracker", info['reward_tracker'])
                self.logger.record(f"trade/seed", info['seed'])
                break
        del trade_model


    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:

        infos = self.locals["infos"]
        sorted_infos = sorted(infos, key=lambda x: (x["counter"], x['portfolio_value']), reverse=True)
        best_info = sorted_infos[0]
        best_info_seed = best_info['seed']

        for k, v in best_info.items():
            if "combined_" in k:
                self.logger.record(f"combined_profit/{k}", v)

            elif "moves" in k:
                self.logger.record(f"moves/{k}", v)

            elif "_loss" in k:
                self.logger.record(f"losses/{k}", v)

            elif "_profit" in k:
                self.logger.record(f"profits/{k}", v)

            elif "_trade" in k or k.endswith("%"):
                self.logger.record(f"trades/{k}", v)

            elif "_counter" in k:
                self.logger.record(f"counters/{k}", v)

            elif "_streak" in k:
                self.logger.record(f"steaks/{k}", v)

            else:
                self.logger.record(f"xcommons/{k}", v)

            # if k == "portfolio_value" and v > 14000:
            #     dir = Path(f"logs/{self.log_name}/")
            #     dir.mkdir(parents=True, exist_ok=True)
            #     (dir / str(self.num_timesteps)).write_text(
            #         json.dumps(best_info, default=str, indent=4)
            #     )

        self.test_and_log(best_info_seed)

    def _on_training_end(self) -> None:
        pass
        # self.test_and_log(1)



def main():
    model_name = f"portfolio_reward_wb_{TICKER.split('.')[0]}_ppo"
    print(model_name)

    check_env(StockTradingEnv(CLOSE_PRICES, seed=SEED))
    vec_env = make_vec_env(
        env_id=StockTradingEnv,
        close_prices=CLOSE_PRICES[:200],
        start_seed=SEED,
        n_envs=NUM_ENVS,
    )

    if Path(f"trained_models/{model_name}.zip").exists():
        reset_num_timesteps = False
        model = PPO.load(
            f"trained_models/{model_name}.zip",
            vec_env,
            print_system_info=True,
            device="auto",
            ent_coef=ENT_COEF,
            n_epochs=N_EPOCHS,
            n_steps=N_STEPS,
            verbose=0
        )
    else:
        reset_num_timesteps = True
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            n_steps=N_STEPS,
            device="auto",
            n_epochs=N_EPOCHS,
            ent_coef=ENT_COEF,
            tensorboard_log="tensorboard_log",
            policy_kwargs = dict(
                net_arch=dict(pi=[1024, 2048, 1024], vf=[1024, 2048, 1024])
            )
        )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
        callback=EvalCallback(),
        tb_log_name=model_name,
    )
    model.save("trained_models/" + model_name)


if __name__ == "__main__":
    main()
