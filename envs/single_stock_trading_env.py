import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


class StockTradingEnv(gym.Env):
    """
    Observations
        [Close_Price, Available_Amount, Shares_Holdings, Buy_Price]
        Close_Price -
        Available_Amount -
        Shares_Holdings -
        Buy_Price -
    Actions
        [HOLD, BUY, SELL]
        [0, 1, 2]
    """

    def __init__(self, close_prices, seed):
        super().__init__()

        self.close_prices = close_prices
        low = np.min(close_prices)
        high = np.max(close_prices) * 10
        self.length = len(self.close_prices)

        self.observation_space = Box(
            low=np.array([0, 0, 0, 0], np.float32),
            high=np.array([high, 30_000, 1000, 30_000], np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = Discrete(3)
        self.seed = seed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=self.seed, options=options)
        self.counter = 0
        self.correct_trade = 0
        self.wrong_trade = 0

        self.buy_counter = 0
        self.sell_counter = 0
        self.hold_counter = 0

        self.good_hold_counter = 0
        self.good_sell_counter = 0
        self.good_buy_counter = 0

        self.bad_buy_counter = 0
        self.bad_hold_counter = 0
        self.bad_sell_counter = 0

        self.holds_with_no_shares_counter = 0

        self.good_hold_profit = 0
        self.good_sell_profit = 0
        self.good_buy_profit = 0

        self.bad_buy_loss = 0
        self.bad_sell_loss = 0
        self.bad_hold_loss = 0

        self.good_profit = 0
        self.bad_loss = 0

        self.good_hold_streak = 0
        self.bad_hold_streak = 0
        self.buy_price_index = -1

        self.reward_tracker = 0

        self.good_moves = 0
        self.bad_moves = 0
        self.track_portfolio = []

        close_price = self.close_prices[self.counter]
        available_amount = 10_000
        shares_holding = 0
        buy_price = 0
        self.state = np.array(
            [close_price, available_amount, shares_holding, buy_price], dtype=np.float32
        )
        return self.state, {}

    def step(self, action):
        reward = 0
        profit = 0
        shares_bought = 0
        shares_sold = 0
        truncated = False
        close_price = self.state[0]
        available_amount = self.state[1]
        shares_holding = self.state[2]
        buy_price = self.state[3]
        description = ""

        predicted_action = ACTION_MAP[action]

        if predicted_action == "BUY":
            self.buy_counter += 1
            if close_price > available_amount:
                reward -= 50_000
                truncated = True
                self.wrong_trade += 1
                description = f"{close_price} > {available_amount}. Cannot Buy Shares"
            else:
                shares_bought = available_amount // close_price

                buy_price = close_price * shares_bought
                shares_holding += shares_bought
                available_amount -= buy_price

                # test simple reward
                reward += 1
                # reward += shares_bought
                # self.track_portfolio.append({
                #     'counter': self.counter,
                #     'buy_price': buy_price,
                #     'shares_holding': shares_holding,
                # })
                self.correct_trade += 1
                self.good_buy_counter += 1
                self.buy_price_index = self.counter
                description = f"{shares_bought} shares bought at {close_price:.2f}"

        elif predicted_action == "SELL":
            self.sell_counter += 1
            if shares_holding == 0:
                reward -= 50_000
                truncated = True
                self.wrong_trade += 1
                description = f"{shares_holding} shares available. Cannot Sell Shares"
            else:
                shares_sold = shares_holding
                sell_price = close_price * shares_holding
                available_amount += sell_price
                profit = sell_price - buy_price
                reward += profit

                if profit > 0:
                    self.good_sell_counter += 1
                    self.good_sell_profit += profit
                else:
                    self.bad_sell_counter += 1
                    self.bad_sell_loss += profit

                # self.track_portfolio.append({
                #     'counter': self.counter,
                #     'sell_price': buy_price,
                #     'shares_sold': shares_sold,
                # })
                shares_holding = 0
                buy_price = 0
                self.correct_trade += 1
                self.good_hold_streak = 0
                self.bad_hold_streak = 0
                self.buy_price_index = -1
                description = f"{shares_sold} shares sold at {close_price:.2f} with profit of {profit}"

        elif predicted_action == "HOLD":
            self.correct_trade += 1
            if shares_holding == 0:
                # test simple reward
                reward += 1
                self.holds_with_no_shares_counter += 1
                description = f"{shares_holding} shares holding."
            else:
                self.hold_counter += 1
                profit = (close_price * shares_holding) - buy_price
                reward += profit

                if profit > 0:
                    h_desc = "GOOD"
                    self.good_hold_counter += 1
                    self.good_hold_profit += profit
                    self.good_hold_streak += 1
                else:
                    self.bad_hold_counter += 1
                    self.bad_hold_loss += profit
                    self.bad_hold_streak += 1
                    h_desc = "BAD"
                description = f"{h_desc} Holding {shares_holding} shares at {buy_price:.2f} profit of {profit}"
        else:
            raise ValueError(f"{action} should be in [0,1,2]")

        if profit > 0:
            self.good_profit += profit
            self.good_moves += 1

        elif profit < 0:
            self.bad_loss += profit
            self.bad_moves += 1

        else:
            # stale_moves
            ...

        self.reward_tracker += reward
        done = self.counter == (self.length - 1)

        correct_trade_percent = self.calculate_percent(
            self.correct_trade, self.correct_trade + self.wrong_trade
        )
        good_sell_counter_percent = self.calculate_percent(
            self.good_sell_counter, self.correct_trade + self.wrong_trade
        )
        good_hold_counter_percent = self.calculate_percent(
            self.good_hold_counter, self.correct_trade + self.wrong_trade
        )

        bad_sell_counter_percent = self.calculate_percent(
            self.bad_sell_counter, self.correct_trade + self.wrong_trade
        )
        bad_hold_counter_percent = self.calculate_percent(
            self.bad_hold_counter, self.correct_trade + self.wrong_trade
        )
        buy_counter_percent = self.calculate_percent(
            self.buy_counter, self.correct_trade + self.wrong_trade
        )

        holds_with_no_shares_counter_percent = self.calculate_percent(
            self.holds_with_no_shares_counter, self.correct_trade + self.wrong_trade
        )

        good_moves_percent = self.calculate_percent(
            self.good_moves, self.good_moves + self.bad_moves
        )

        combined_hold_profit = self.good_hold_profit + self.bad_hold_loss
        combined_sell_profit = self.good_sell_profit + self.bad_sell_loss

        portfolio_value = shares_holding * close_price + available_amount
        info = {
            "seed": self.seed,
            "counter": self.counter,
            "close_price": close_price,
            "predicted_action": predicted_action,
            "description": description,
            "available_amount": available_amount,
            "shares_holdings": shares_holding,
            "buy_price": buy_price,
            "buy_price_index": self.buy_price_index,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "correct_trade": self.correct_trade,
            "wrong_trade": self.wrong_trade,
            "correct_trade %": correct_trade_percent,
            "buy_counter": self.buy_counter,
            "sell_counter": self.sell_counter,
            "hold_counter": self.hold_counter,
            "good_hold_counter": self.good_hold_counter,
            "good_sell_counter": self.good_sell_counter,
            "good_buy_counter": self.good_buy_counter,
            "bad_hold_counter": self.bad_hold_counter,
            "bad_sell_counter": self.bad_sell_counter,
            "bad_buy_counter": self.bad_buy_counter,
            "hold_with_no_shares_counter": self.holds_with_no_shares_counter,
            "good_hold_streak": self.good_hold_streak,
            "bad_hold_streak": self.bad_hold_streak,
            "buy_counter %": buy_counter_percent,
            "good_sell_counter %": good_sell_counter_percent,
            "good_hold_counter %": good_hold_counter_percent,
            "bad_sell_counter %": bad_sell_counter_percent,
            "bad_hold_counter %": bad_hold_counter_percent,
            "holds_with_no_shares_counter %": holds_with_no_shares_counter_percent,
            "good_hold_profit": self.good_hold_profit,
            "good_sell_profit": self.good_sell_profit,
            "good_buy_profit": self.good_buy_profit,
            "bad_hold_loss": self.bad_hold_loss,
            "bad_sell_loss": self.bad_sell_loss,
            "bad_buy_loss": self.bad_buy_loss,
            "good_moves": self.good_moves,
            "bad_moves": self.bad_moves,
            "good_moves %": good_moves_percent,
            "reward_tracker": self.reward_tracker,
            "portfolio_value": portfolio_value,
            "combined_hold_profit": combined_hold_profit,
            "combined_sell_profit": combined_sell_profit,
            "combined_total_profit": combined_hold_profit + combined_sell_profit,
            "combined_hold_streak": self.good_hold_streak + self.bad_hold_streak,
            "track_portfolio": self.track_portfolio,
        }

        if done or truncated:
            return self.state, float(reward), done, truncated, info

        self.counter += 1
        close_price = self.close_prices[self.counter]
        self.state = np.array(
            [close_price, available_amount, shares_holding, buy_price], dtype=np.float32
        )
        return self.state, float(reward), done, truncated, info

    def close(self):
        pass

    def calculate_percent(self, v1, v2):
        if v1 == 0 or v2 == 0:
            return 0
        return round((v1 / (v2)) * 100, 2)
