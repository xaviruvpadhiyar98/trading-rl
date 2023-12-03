import json
from pathlib import Path

import polars as pl
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from common.make_vec_env import make_vec_env
from envs.single_stock_trading_env import StockTradingEnv

TICKER = "SBIN.NS"
EVAL_FILE = Path("datasets") / f"{TICKER}_trade"
EVAL_CLOSE_PRICES = pl.read_parquet(EVAL_FILE)["Close"].to_numpy()


class EvalCallback(BaseCallback):
    def __init__(self, log_counter=64000, model_name="a2c"):
        super().__init__()
        self.log_counter = log_counter
        self.model_name = model_name

    def _on_training_start(self) -> None:
        self.test_and_log()

    def _on_rollout_start(self) -> None:
        pass

    def test_and_log(self) -> None:
        eval_vec_env = make_vec_env(
            env_id=StockTradingEnv,
            close_prices=EVAL_CLOSE_PRICES,
            start_seed=1337,
            n_envs=128,
        )

        model = {"ppo": PPO, "a2c": A2C}[self.model_name]

        trade_model = model("MlpPolicy", eval_vec_env)
        trade_model.set_parameters(self.model.get_parameters())
        episode_rewards, episode_lengths = evaluate_policy(
            trade_model, eval_vec_env, n_eval_episodes=1, return_episode_rewards=True
        )
        self.logger.record(f"trade/ep_len", episode_lengths[0])
        self.logger.record(f"trade/ep_reward", episode_rewards[0])

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:

        if (((self.num_timesteps % self.log_counter) != 0) and self.model_name == "a2c"):
            return True

        infos = self.locals["infos"]
        sorted_infos = sorted(
            infos, key=lambda x: x["portfolio_value"], reverse=True
        )
        best_info = sorted_infos[0]

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
                self.logger.record(f"commons/{k}", v)

            if k == "portfolio_value" and v > 18000:
                dir = Path(f"logs/{self.model_name}/")
                dir.mkdir(parents=True, exist_ok=True)
                (dir / str(self.num_timesteps)).write_text(
                    json.dumps(best_info, default=str, indent=4)
                )

        self.test_and_log()

    def _on_training_end(self) -> None:
        self.test_and_log()
