import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from math import exp

ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


class StockTradingEnv(gym.Env):
    """
    Observations
        Close_Price - Past 40 Prices
        Available_Amount -
        Shares_Holdings -
        Buy_Price -
        Profit - 0
        Portfolio - 0
    Actions
        [HOLD, BUY, SELL]
        [0, 1, 2]
    """

    def __init__(self, close_prices, seed):
        super().__init__()

        self.close_prices = close_prices
        # low = np.min(close_prices)
        # high = np.max(close_prices) * 10
        self.length = len(self.close_prices)

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(40 + 5,),
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
        self.bad_holds_with_no_shares_counter = 0
        self.good_holds_with_no_shares_counter = 0

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
        self.waiting_streak = 0
        self.buy_price_index = -1

        self.reward_tracker = 0

        self.good_moves = 0
        self.bad_moves = 0
        self.track_portfolio = []

        past_n_prices = self.close_prices[self.counter]
        close_price = past_n_prices[-1]
        available_amount = 10_000
        shares_holding = 0
        buy_price = 0
        profit = 0
        portfolio_value = available_amount
        self.state = np.concatenate(
            (
                past_n_prices,
                np.array(
                    (
                        available_amount,
                        shares_holding,
                        buy_price,
                        profit,
                        portfolio_value,
                    )
                ),
            ),
            dtype=np.float32,
        )
        return self.state, {}

    def step(self, action):
        reward = 0
        profit = 0
        shares_bought = 0
        shares_sold = 0
        truncated = False
        past_n_prices = self.state[:-5]
        close_price = past_n_prices[-1]
        available_amount = self.state[-5]
        shares_holding = self.state[-4]
        buy_price = self.state[-3]
        profit = self.state[-2]
        portfolio_value = self.state[-1]

        min_price = min(past_n_prices)
        max_price = max(past_n_prices)
        average_price = sum(past_n_prices) / len(past_n_prices)

        percentage_hold_threshold = 10
        percentage_buy_threshold = 5
        percentage_sell_threshold = 5

        percentage_distance_from_min = (
            (close_price - min_price) / (max_price - min_price)
        ) * 100
        percentage_distance_from_max = 100 - percentage_distance_from_min
        closer_to_min_price_with_threshold = (
            percentage_buy_threshold > percentage_distance_from_min
        )

        buy_range_value = min_price * (percentage_buy_threshold / 100)
        sell_range_value = max_price * (percentage_sell_threshold / 100)
        hold_range_value = average_price * (percentage_hold_threshold / 100)

        is_near_min_good_buy = (
            min_price - buy_range_value <= close_price <= min_price + buy_range_value
        )

        # is_near_max = max_price - sell_range_value <= close_price <= max_price + sell_range_value
        # is_near_hold = max_past_n_prices - hold_range_value <= close_price <= max_past_n_prices + hold_range_value

        # holding_threshold = average_past_n_prices * (1 - percentage_below_average / 100)

        predicted_action = ACTION_MAP[action]

        # Attempting to buy without sufficient funds.
        if predicted_action == "BUY" and close_price > available_amount:
            reward -= 50_000
            truncated = True
            self.wrong_trade += 1
            description = (
                f"[!] Transaction #{self.counter}: Purchase Denied. "
                f"Reason: Insufficient Funds. "
                f"Attempted Purchase Price: ₹{close_price} "
                f"Available Funds: ₹{available_amount}"
            )

        # Attempting to sell when no shares are held.
        elif predicted_action == "SELL" and shares_holding == 0:
            reward -= 50_000
            truncated = True
            self.wrong_trade += 1
            description = (
                f"[!] Transaction #{self.counter}: Sale Denied. "
                f"Reason: Insufficient Shares. "
                f"Attempted to Sell at Price: ₹{close_price}."
            )

        # Holding when there are no shares and a good buying opportunity is present.
        # elif (
        #     predicted_action == "HOLD"
        #     and shares_holding == 0
        #     and min_past_n_prices > close_price
        # ):
        #     reward -= 50_000
        #     truncated = True
        #     self.wrong_trade += 1
        #     description = (
        #         f"[!] Transaction #{self.counter}: Missed Opportunity. "
        #         f"Reason: Good Buying Opportunity Missed While Holding No Shares. "
        #         f"Close Price: ₹{close_price} was within ₹{lower_bound} - ₹{upper_bound}, "
        #         f"indicative of a potential buy."
        #     )

        # elif predicted_action == "SELL" and shares_holding > 0 and self.bad_sell_counter > 3:
        #     reward -= 50_000
        #     truncated = True
        #     self.wrong_trade += 1
        #     description = (
        #         f"[!] Transaction #{self.counter}: Keeps Selling at Loss. "
        #         f"Reason: Occured Loss {self.bad_sell_counter} times."
        #         f"Attempted to Sell at Price: ₹{close_price} "
        #         f"Available Shares for Sale: {shares_holding}"
        #     )

        # elif predicted_action == "HOLD" and (max_past_n_prices - min_past_n_prices) > 15:
        #     reward -= 50_000
        #     truncated = True
        #     self.wrong_trade += 1
        #     current_holdings = close_price*shares_holding
        #     loss = buy_price - current_holdings

        #     description = (
        #         f"[!] Transaction #{self.counter}: Missed Opportunity. "
        #         f"Reason: Could have bought at ₹{min_past_n_prices} and sold at ₹{max_past_n_prices}. "
        #         f"Shares Held: {shares_holding} "
        #         f"Purchase Price:₹{buy_price:.2f} "
        #         f"Current Price: ₹{current_holdings:.2f} "
        #         f"Loss: {loss}"
        #     )
        # elif predicted_action == "HOLD" and self.holds_with_no_shares_counter > 10:
        #     reward -= 50_000
        #     truncated = True
        #     self.wrong_trade += 1
        #     current_holdings = close_price*shares_holding
        #     loss = buy_price - current_holdings

        #     description = (
        #         f"[!] Transaction #{self.counter}: Bad Hold Steak. "
        #         f"Reason: {self.bad_hold_streak} consecutive bad holds. "
        #         f"Shares Held: {shares_holding} "
        #         f"Purchase Price:₹{buy_price:.2f} "
        #         f"Current Price: ₹{current_holdings:.2f} "
        #         f"Loss: {loss}"
        #     )

        elif (
            predicted_action == "BUY"
            and close_price <= available_amount
            and closer_to_min_price_with_threshold
        ):
            shares_bought = available_amount // close_price
            buy_price = close_price
            shares_holding += shares_bought
            available_amount -= buy_price * shares_bought
            reward += percentage_distance_from_max
            description = (
                f"[+] Transaction #{self.counter}: Good Purchase Successful. "
                f"Shares Acquired: {shares_bought}. "
                f"Purchase Price per Share: ₹{close_price:.2f}. "
                f"Total Buying Price: ₹{buy_price}. "
                f"Available Amount: ₹{available_amount}. "
                f"Waiting Period Before Purchase: {self.waiting_streak} intervals. "
                f"Reward Earned: ₹{reward} (Good). "
                f"Average Price {average_price} is less than Buying Price {buy_price} "
                f"Minimum Price {min_price} Maximum Price {max_price} "
                f"Portfolio Value: ₹{portfolio_value}"
            )
            self.buy_price_index = self.counter
            self.buy_counter += 1
            self.good_hold_streak = 0
            self.bad_hold_streak = 0
            self.waiting_streak = 0
            self.correct_trade += 1
            self.good_buy_counter += 1

        elif (
            predicted_action == "BUY"
            and close_price <= available_amount
            and not closer_to_min_price_with_threshold
        ):
            shares_bought = available_amount // close_price
            buy_price = close_price
            shares_holding += shares_bought
            available_amount -= buy_price * shares_bought
            reward += min_price - close_price
            description = (
                f"[+] Transaction #{self.counter}: Bad Purchase Successful. "
                f"Shares Acquired: {shares_bought}. "
                f"Purchase Price per Share: ₹{close_price:.2f}. "
                f"Total Buying Price: ₹{buy_price}. "
                f"Available Amount: ₹{available_amount}. "
                f"Waiting Period Before Purchase: {self.waiting_streak} intervals. "
                f"Reward Earned: ₹{reward} (BAD). "
                f"Average Price {average_price} is less than Buying Price {buy_price} "
                f"Minimum Price {min_price} Maximum Price {max_price} "
                f"Portfolio Value: ₹{portfolio_value}"
            )
            self.buy_price_index = self.counter
            self.buy_counter += 1
            self.good_hold_streak = 0
            self.bad_hold_streak = 0
            self.waiting_streak = 0
            self.correct_trade += 1
            self.bad_buy_counter += 1

        elif (
            predicted_action == "SELL"
            and shares_holding > 0
            and close_price > buy_price
        ):
            shares_sold = shares_holding
            sell_price = close_price * shares_holding
            available_amount += sell_price
            profit = sell_price - buy_price * shares_holding
            reward += profit
            portfolio_value = available_amount

            description = (
                f"[+] Transaction #{self.counter}: Profitable Sale Executed. "
                f"Shares Sold: {shares_sold}. "
                f"Purchase Price per Share: ₹{buy_price}. "
                f"Sale Price per Share: ₹{close_price}. "
                f"Profit Earned: ₹{profit}. "
                f"Holding Performance: Good Streak - {self.good_hold_streak}, Bad Streak - {self.bad_hold_streak}. "
                f"Holding Performance2: Good Hold - {self.good_hold_counter}, Bad Hold - {self.bad_hold_counter}. "
                f"Reward Earned: ₹{reward}. "
                f"Portfolio Value: ₹{portfolio_value}."
            )

            self.good_sell_counter += 1
            self.sell_counter += 1
            self.good_sell_profit += profit
            shares_holding = 0
            buy_price = 0
            self.good_hold_streak = 0
            self.bad_hold_streak = 0
            self.buy_price_index = -1

        elif (
            predicted_action == "SELL"
            and shares_holding > 0
            and close_price <= buy_price
        ):
            shares_sold = shares_holding
            sell_price = close_price * shares_holding
            available_amount += sell_price
            profit = sell_price - buy_price * shares_holding
            reward += profit
            portfolio_value = available_amount

            description = (
                f"[+] Transaction #{self.counter}: Sale Executed with Loss. "
                f"Shares Sold: {shares_sold}. "
                f"Purchase Price per Share: ₹{buy_price}. "
                f"Sale Price per Share: ₹{close_price}. "
                f"Loss Earned: ₹{profit}. "
                f"Holding Performance: Good Streak - {self.good_hold_streak}, Bad Streak - {self.bad_hold_streak}. "
                f"Holding Performance2: Good Hold - {self.good_hold_counter}, Bad Hold - {self.bad_hold_counter}. "
                f"Reward Earned: ₹{reward}. "
                f"Portfolio Value: ₹{portfolio_value}."
            )

            self.bad_sell_counter += 1
            self.sell_counter += 1
            self.bad_sell_loss += profit
            shares_holding = 0
            buy_price = 0
            self.good_hold_streak = 0
            self.bad_hold_streak = 0
            self.buy_price_index = -1

        elif (
            predicted_action == "HOLD"
            and shares_holding == 0
            and closer_to_min_price_with_threshold
        ):
            # reward += close_price - holding_threshold
            # reward *= 1000
            reward += min_price - close_price
            if reward < 20:
                reward = -600
            description = (
                f"[·] Transaction #{self.counter}: Missed Buying Opportunity. "
                f"Duration of Waiting: {self.waiting_streak} intervals. "
                f"Current Share Price: ₹{close_price:.2f}. "
                f"Average Price {average_price} is less than Buying Price {buy_price} "
                f"Minimum Price {min_price} Maximum Price {max_price} "
                f"Reward Accumulated: ₹{reward}. "
                f"Portfolio Value: ₹{portfolio_value}."
            )
            self.waiting_streak += 1
            self.holds_with_no_shares_counter += 1
            self.bad_holds_with_no_shares_counter += 1

        elif (
            predicted_action == "HOLD"
            and shares_holding == 0
            and not closer_to_min_price_with_threshold
        ):
            reward += percentage_distance_from_max
            description = (
                f"[·] Transaction #{self.counter}: Correct decision to HOLD. "
                f"Duration of Waiting: {self.waiting_streak} intervals. "
                f"Current Share Price: ₹{close_price:.2f}. "
                f"Average Price {average_price} is less than Buying Price {buy_price} "
                f"Minimum Price {min_price} Maximum Price {max_price} "
                f"Reward Accumulated: ₹{reward}. "
                f"Portfolio Value: ₹{portfolio_value}."
            )
            self.waiting_streak += 1
            self.holds_with_no_shares_counter += 1
            self.good_holds_with_no_shares_counter += 1

        elif (
            predicted_action == "HOLD"
            and shares_holding > 0
            and close_price >= buy_price
        ):
            profit = (close_price - buy_price) * shares_holding
            reward += profit
            portfolio_value = close_price * shares_holding + available_amount
            description = (
                f"[+] Transaction #{self.counter}: Profitable Holding. "
                f"Shares Held: {shares_holding}. "
                f"Purchase Price per Share: ₹{buy_price:.2f}. "
                f"Current Share Price: ₹{close_price:.2f}. "
                f"Unrealized Profit: ₹{profit}. "
                f"Holding Performance: Good Streak - {self.good_hold_streak}, Bad Streak - {self.bad_hold_streak}. "
                f"Holding Performance2: Good Hold - {self.good_hold_counter}, Bad Hold - {self.bad_hold_counter}. "
                f"Reward Earned: ₹{reward}. "
                f"Portfolio Value: ₹{portfolio_value}."
            )
            self.good_hold_counter += 1
            self.good_hold_profit += profit
            self.good_hold_streak += 1
            self.bad_hold_streak = 0
            self.hold_counter += 1
            self.waiting_streak = 0

        elif (
            predicted_action == "HOLD"
            and shares_holding > 0
            and close_price < buy_price
        ):
            profit = (close_price - buy_price) * shares_holding
            reward += profit
            reward *= 10
            portfolio_value = close_price * shares_holding + available_amount
            description = (
                f"[+] Transaction #{self.counter}: Unprofitable Holding. "
                f"Shares Held: {shares_holding}. "
                f"Purchase Price per Share: ₹{buy_price:.2f}. "
                f"Current Share Price: ₹{close_price:.2f}. "
                f"Unrealized Loss: ₹{profit} (Negative). "
                f"Holding Performance: Good Streak - {self.good_hold_streak}, Bad Streak - {self.bad_hold_streak}. "
                f"Holding Performance2: Good Hold - {self.good_hold_counter}, Bad Hold - {self.bad_hold_counter}. "
                f"Reward Earned: ₹{reward}. "
                f"Portfolio Value: ₹{portfolio_value}."
            )
            self.bad_hold_counter += 1
            self.bad_hold_loss += profit
            self.bad_hold_streak += 1
            self.good_hold_streak = 0
            self.hold_counter += 1
            self.waiting_streak = 0
        else:
            ...

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
            "bad_holds_with_no_shares_counter": self.bad_holds_with_no_shares_counter,
            "good_holds_with_no_shares_counter": self.good_holds_with_no_shares_counter,
            "good_hold_streak": self.good_hold_streak,
            "bad_hold_streak": self.bad_hold_streak,
            "waiting_streak": self.waiting_streak,
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
        past_n_prices = self.close_prices[self.counter]
        self.state = np.concatenate(
            (
                past_n_prices,
                np.array(
                    (
                        available_amount,
                        shares_holding,
                        buy_price,
                        profit,
                        portfolio_value,
                    )
                ),
            ),
            dtype=np.float32,
        )
        return self.state, float(reward), done, truncated, info

    def close(self):
        pass

    def calculate_percent(self, v1, v2):
        if v1 == 0 or v2 == 0:
            return 0
        return round((v1 / (v2)) * 100, 2)
