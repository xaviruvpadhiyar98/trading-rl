import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


class StockTradingEnv(gym.Env):
    """
    Observations
        Close_Price - Past 40 Prices
        Available_Amount -
        Shares_Holdings -
        Buy_Price -
        total_buy_price -
        Profit - 0
        Portfolio - 0
    Actions
        [HOLD, BUY, SELL]
        [0, 1, 2]
    """

    def __init__(self, close_prices, seed):
        super().__init__()

        self.close_prices = close_prices
        self.length = len(self.close_prices)

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(40 + 7,),
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
        total_buy_price = buy_price * shares_holding
        profit = 0
        portfolio_value = available_amount
        portfolio_value_threshold = portfolio_value
        self.state = np.concatenate(
            (
                past_n_prices,
                np.array(
                    (
                        available_amount,
                        shares_holding,
                        buy_price,
                        total_buy_price,
                        profit,
                        portfolio_value,
                        portfolio_value_threshold,
                    )
                ),
            ),
            dtype=np.float32,
        )
        return self.state, {}

    def log(
        self,
        short_desc,
        shares_holding,
        close_price,
        available_amount,
        portfolio_value,
        reward,
        buy_price,
        min_price,
        max_price,
    ):
        description = (
            f"[+] {self.predicted_action} Transaction #{self.counter}: {short_desc}. <br>"
            f"Min Price: {min_price:.2f}. <br> "
            f"Close Price: {close_price:.2f}. <br> "
            f"Max Price: {max_price:.2f}. <br> "
            f"Reward: {reward}. <br> "
            f"Reward Tracker: {self.reward_tracker}. <br> "
            f"Shares: {shares_holding}. <br> "
            f"Available Amount: {available_amount}. <br> "
            f"Portfolio Value: {portfolio_value}. <br> "
            f"Buy Counters: {self.buy_counter}. <br> "
            f"&emsp; Good Buy Counters: {self.good_buy_counter}. <br> "
            f"&emsp; BAD Buy Counters: {self.bad_buy_counter}. <br> "
            f"Sell Counters: {self.sell_counter}. <br> "
            f"&emsp; GOOD SELL Counters: {self.good_sell_counter}. <br> "
            f"&emsp; BAD SELL Counters: {self.bad_sell_counter}. <br> "
            f"Hold Counters: {self.hold_counter}. <br> "
            f"&emsp; GOOD HOLD Counters: {self.good_hold_counter}. <br> "
            f"&emsp; BAD HOLD Counters: {self.bad_hold_counter}. <br> "
            f"Waiting Period Before Purchase: {self.waiting_streak} intervals. <br> "
            f"Good Streak: {self.good_hold_streak}. <br>"
            f"Bad Streak: {self.bad_hold_streak}. <br>"
            f"Holding with No Shares Counter: {self.holds_with_no_shares_counter} intervals. <br> "
            f"&emsp; GOOD HOLD with No Shares Counters: {self.good_holds_with_no_shares_counter}. <br> "
            f"&emsp; BAD HOLD with No Shares Counters: {self.bad_holds_with_no_shares_counter}. <br> "
            f"Buying Price: ₹{buy_price}. <br> "
        )
        return description

    def step(self, action):
        reward = 0
        profit = 0
        shares_bought = 0
        shares_sold = 0
        truncated = False
        past_n_prices = self.state[:-7]
        close_price = past_n_prices[-1]
        available_amount = self.state[-7]
        shares_holding = self.state[-6]
        buy_price = self.state[-5]
        total_buy_price = self.state[-4]
        profit = self.state[-3]
        portfolio_value = self.state[-2]
        portfolio_value_threshold = self.state[-1]

        min_price = min(past_n_prices)
        max_price = max(past_n_prices)
        average_price = sum(past_n_prices) / len(past_n_prices)

        percentage_hold_threshold = 10
        percentage_buy_threshold = 30
        percentage_sell_threshold = 5

        percentage_distance_from_min = (
            (close_price - min_price) / (max_price - min_price)
        ) * 100
        percentage_distance_from_max = 100 - percentage_distance_from_min
        closer_to_min_price_with_threshold = (
            percentage_distance_from_min < percentage_buy_threshold
        )

        buy_range_value = min_price * (percentage_buy_threshold / 100)
        sell_range_value = max_price * (percentage_sell_threshold / 100)
        hold_range_value = average_price * (percentage_hold_threshold / 100)

        is_near_min_good_buy = (
            min_price - buy_range_value <= close_price <= min_price + buy_range_value
        )

        predicted_action = ACTION_MAP[action]
        self.predicted_action = predicted_action

        # Attempting to buy without sufficient funds.
        if predicted_action == "BUY" and close_price > available_amount:
            reward -= 50_000
            truncated = True
            self.wrong_trade += 1
            short_desc = "BUY without funds"
            description = self.log(
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                buy_price,
                min_price,
                max_price,
            )
            # description = (
            #     f"[!] Transaction #{self.counter}: Purchase Denied. "
            #     f"Reason: Insufficient Funds. "
            #     f"Attempted Purchase Price: ₹{close_price} "
            #     f"Available Funds: ₹{available_amount}. "
            # )

        # Attempting to sell when no shares are held.
        elif predicted_action == "SELL" and shares_holding == 0:
            reward -= 50_000
            truncated = True
            self.wrong_trade += 1
            short_desc = "SELL without Shares"
            description = self.log(
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                buy_price,
                min_price,
                max_price,
            )
            # description = (
            #     f"[!] Transaction #{self.counter}: Sale Denied. "
            #     f"Reason: Insufficient Shares. "
            #     f"Attempted to Sell at Price: ₹{close_price}."
            # )

        # Attempting to sell when no shares are held.
        # elif predicted_action == "HOLD" and self.waiting_streak > 100:
        #     reward -= 50_000
        #     truncated = True
        #     self.wrong_trade += 1
        #     short_desc = "Keeps HOLDING without shares"
        #     description = self.log(
        #         short_desc,
        #         shares_holding,
        #         close_price,
        #         available_amount,
        #         portfolio_value,
        #         reward,
        #         buy_price,
        #         min_price,
        #         max_price,
        #     )
            # description = (
            #     f"[!] Transaction #{self.counter}: Waited too many times. "
            #     f"Waited {self.waiting_streak}"
            #     f"Good Waiting: {self.good_holds_with_no_shares_counter}. "
            #     f"BAD Waiting: {self.bad_holds_with_no_shares_counter}."
            # )
        # Attempting to sell when no shares are held.
        # elif predicted_action == "HOLD" and self.bad_hold_counter > 10:
        #     reward -= 50_000
        #     truncated = True
        #     self.wrong_trade += 1
        #     short_desc = "BAD HOLD COUNTER > 10"
        #     description = self.log(
        #         short_desc,
        #         shares_holding,
        #         close_price,
        #         available_amount,
        #         portfolio_value,
        #         reward,
        #         buy_price,
        #         min_price,
        #         max_price,
        #     )
            # description = (
            #     f"[!] Transaction #{self.counter}: BAD HOLD more than 10 times "
            #     f"Waited {self.waiting_streak}"
            #     f"Good Waiting: {self.good_holds_with_no_shares_counter}. "
            #     f"BAD Waiting: {self.bad_holds_with_no_shares_counter}."
            # )
        # Attempting to sell when no shares are held.
        elif predicted_action == "SELL" and (
            self.good_hold_streak < 2 or self.bad_hold_streak < 2
        ):
            reward -= 50_000
            truncated = True
            self.wrong_trade += 1
            short_desc = "SELLING DIRECT AFTER BUY"
            description = self.log(
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                buy_price,
                min_price,
                max_price,
            )
            # description = (
            #     f"[!] Transaction #{self.counter}: DIRECT SELL after buy "
            #     f"Waited {self.waiting_streak}"
            #     f"Good Waiting: {self.good_holds_with_no_shares_counter}. "
            #     f"BAD Waiting: {self.bad_holds_with_no_shares_counter}."
            # )

        elif predicted_action == "BUY" and close_price <= available_amount:
            shares_bought = available_amount // close_price
            buy_price = close_price
            shares_holding += shares_bought
            total_buy_price = buy_price * shares_bought
            available_amount -= total_buy_price

            if percentage_distance_from_max <= percentage_buy_threshold:
                self.good_buy_counter += 1
                short_desc = "GOOD"
                reward += (
                    100
                    - (percentage_distance_from_max / percentage_buy_threshold) * 100
                )
            else:
                self.bad_buy_counter += 1
                short_desc = "BAD"
                reward += (
                    -(
                        (percentage_distance_from_max - percentage_buy_threshold)
                        / percentage_buy_threshold
                    )
                    * 100
                )

            short_desc = f"{short_desc} Purchase Successful"
            description = self.log(
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                buy_price,
                min_price,
                max_price,
            )

            # description = (
            #     f"[+] Transaction #{self.counter}: {short_desc} Purchase Successful. "
            #     f"Shares Acquired: {shares_bought}. "
            #     f"Purchase Price per Share: ₹{close_price:.2f}. "
            #     f"Total Buying Price: ₹{total_buy_price}. "
            #     f"Available Amount: ₹{available_amount}. "
            #     f"Waiting Period Before Purchase: {self.waiting_streak} intervals. "
            #     f"Reward Earned: ₹{reward} (Good). "
            #     f"Minimum Price {min_price} Maximum Price {max_price} "
            #     f"Portfolio Value: ₹{portfolio_value}"
            # )
            # self.track_portfolio.append({
            #     'counter': self.counter,
            #     'buy_price': buy_price,
            #     'shares_holding': shares_holding,
            # })

            self.buy_price_index = self.counter
            self.buy_counter += 1
            self.good_hold_streak = 0
            self.bad_hold_streak = 0
            self.waiting_streak = 0
            self.correct_trade += 1

        elif predicted_action == "SELL" and shares_holding > 0:
            shares_sold = shares_holding
            sell_price = close_price
            total_sell_price = sell_price * shares_sold
            available_amount += total_sell_price
            profit = total_sell_price - total_buy_price
            reward += profit
            portfolio_value = available_amount

            if portfolio_value > portfolio_value_threshold:
                short_desc = "Profitable"
                self.good_sell_counter += 1
                self.good_sell_profit += profit
                portfolio_value_threshold = portfolio_value
            else:
                short_desc = "UnProfitable"
                self.bad_sell_counter += 1
                self.bad_sell_loss += profit
                reward *= 10
            
            short_desc = f"{short_desc} Sale Executed."
            description = self.log(
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                buy_price,
                min_price,
                max_price,
            )
            # description = (
            #     f"[+] Transaction #{self.counter}: {short_desc}  "
            #     f"Shares Sold: {shares_sold}. "
            #     f"Purchase Price per Share: ₹{buy_price}. "
            #     f"Sale Price per Share: ₹{close_price}. "
            #     f"Profit Earned: ₹{profit}. "
            #     f"Holding Performance: Good Streak - {self.good_hold_streak}, Bad Streak - {self.bad_hold_streak}. "
            #     f"Reward Earned: ₹{reward}. "
            #     f"Portfolio Value: ₹{portfolio_value}."
            #     f"Portfolio Value Threshold: ₹{portfolio_value_threshold}."
            # )
            # self.track_portfolio.append({
            #     'counter': self.counter,
            #     'sell_price': sell_price,
            #     'shares_holding': shares_sold,
            # })
            self.sell_counter += 1
            shares_holding = 0
            buy_price = 0
            total_buy_price = 0
            self.good_hold_streak = 0
            self.bad_hold_streak = 0

        elif predicted_action == "HOLD":
            if shares_holding == 0:
                if percentage_distance_from_min <= percentage_buy_threshold:
                    reward += (
                        100
                        - (percentage_distance_from_min / percentage_buy_threshold)
                        * 100
                    )
                    # reward *= 10
                    short_desc = "MISSED Buying Opportunity"
                    self.bad_holds_with_no_shares_counter += 1
                else:
                    reward += (
                        -(
                            (percentage_distance_from_min - percentage_buy_threshold)
                            / percentage_buy_threshold
                        )
                        * 100
                    )
                    short_desc = "Waiting for Good Opportunity"
                    self.good_holds_with_no_shares_counter += 1

                short_desc = f"{short_desc}"
                description = self.log(
                    short_desc,
                    shares_holding,
                    close_price,
                    available_amount,
                    portfolio_value,
                    reward,
                    buy_price,
                    min_price,
                    max_price,
                )
                # description = (
                #     f"[·] Transaction #{self.counter}: {short_desc}. "
                #     f"Duration of Waiting: {self.waiting_streak} intervals. "
                #     f"Current Share Price: ₹{close_price:.2f}. "
                #     f"Minimum Price {min_price} Maximum Price {max_price} "
                #     f"Reward Accumulated: ₹{reward}. "
                #     f"Portfolio Value: ₹{portfolio_value}."
                # )
                self.waiting_streak += 1
                self.holds_with_no_shares_counter += 1
            else:
                portfolio_value = shares_holding * close_price + available_amount
                if portfolio_value > portfolio_value_threshold:
                    profit = portfolio_value - portfolio_value_threshold
                    reward += profit
                    short_desc = "Profitable Holding"
                    self.good_hold_counter += 1
                    self.good_hold_profit += profit
                    self.good_hold_streak += 1
                    self.bad_hold_streak = 0
                else:
                    profit = portfolio_value - portfolio_value_threshold
                    reward += profit
                    short_desc = "UnProfitable Holding"
                    self.bad_hold_counter += 1
                    self.bad_hold_loss += profit
                    self.bad_hold_streak += 1
                    self.good_hold_streak = 0

                short_desc = f"{short_desc}"
                description = self.log(
                    short_desc,
                    shares_holding,
                    close_price,
                    available_amount,
                    portfolio_value,
                    reward,
                    buy_price,
                    min_price,
                    max_price,
                )

                # description = (
                #     f"[+] Transaction #{self.counter}: {short_desc}. "
                #     f"Shares Held: {shares_holding}. "
                #     f"Purchase Price per Share: ₹{buy_price:.2f}. "
                #     f"Current Share Price: ₹{close_price:.2f}. "
                #     f"Profit/Loss: ₹{profit} "
                #     f"Holding Performance: Good Streak - {self.good_hold_streak}, Bad Streak - {self.bad_hold_streak}. "
                #     f"Reward Earned: ₹{reward}. "
                #     f"Portfolio Value: ₹{portfolio_value}."
                #     f"Portfolio Value Threshold: ₹{portfolio_value_threshold}."
                # )
                self.hold_counter += 1
                self.waiting_streak = 0
        else:
            print(predicted_action)
            ValueError("Something is wrong in conditions")

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
            "min_price": min_price,
            "max_price": max_price,
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
            "percentage_distance_from_min": percentage_distance_from_min,
            "percentage_distance_from_max": percentage_distance_from_max,
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
            "portfolio_value_threshold": portfolio_value_threshold,
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
                        total_buy_price,
                        profit,
                        portfolio_value,
                        portfolio_value_threshold,
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
