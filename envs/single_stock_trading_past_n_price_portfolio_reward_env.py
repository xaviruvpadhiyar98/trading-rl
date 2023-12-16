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
            shape=(77,),
            dtype=np.float32,
        )
        self.action_space = Discrete(3)
        self.seed = seed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=self.seed, options=options)

        counter = 0
        correct_trade = 0
        wrong_trade = 0

        buy_counter = 0
        sell_counter = 0
        hold_counter = 0

        good_hold_counter = 0
        good_sell_counter = 0
        good_buy_counter = 0

        bad_buy_counter = 0
        bad_hold_counter = 0
        bad_sell_counter = 0

        holds_with_no_shares_counter = 0
        bad_holds_with_no_shares_counter = 0
        good_holds_with_no_shares_counter = 0

        good_hold_profit = 0
        good_sell_profit = 0
        good_buy_profit = 0

        bad_buy_loss = 0
        bad_sell_loss = 0
        bad_hold_loss = 0

        good_profit = 0
        bad_loss = 0

        good_hold_streak = 0
        bad_hold_streak = 0
        waiting_streak = 0

        reward_tracker = 0

        past_n_prices = self.close_prices[counter]
        close_price = past_n_prices[-1]

        min_price = min(past_n_prices)
        max_price = max(past_n_prices)
        average_price = sum(past_n_prices) / len(past_n_prices)

        percentage_distance_from_min = (
            (close_price - min_price) / (max_price - min_price)
        ) * 100
        percentage_distance_from_max = 100 - percentage_distance_from_min

        available_amount = 10_000
        shares_holding = 0
        buy_price_index = -1
        buy_price = 0
        total_buy_price = buy_price * shares_holding
        profit = 0
        portfolio_value = available_amount
        portfolio_value_threshold = portfolio_value


        self.state = np.concatenate(
            (
                np.array([counter]),
                past_n_prices,
                np.array(
                    (
                        percentage_distance_from_min,
                        percentage_distance_from_max,
                        correct_trade,
                        wrong_trade,
                        buy_counter,
                        sell_counter,
                        hold_counter,
                        good_buy_counter,
                        good_sell_counter,
                        good_hold_counter,
                        bad_buy_counter,
                        bad_sell_counter,
                        bad_hold_counter,
                        holds_with_no_shares_counter,
                        good_holds_with_no_shares_counter,
                        bad_holds_with_no_shares_counter,
                        good_hold_profit,
                        good_sell_profit,
                        good_buy_profit,
                        bad_buy_loss,
                        bad_sell_loss,
                        bad_hold_loss,
                        good_profit,
                        bad_loss,
                        good_hold_streak,
                        bad_hold_streak,
                        waiting_streak,
                        buy_price_index,
                        reward_tracker
                    )
                ),
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
        predicted_action,
        counter,
        short_desc,
        shares_holding,
        close_price,
        available_amount,
        portfolio_value,
        reward,
        reward_tracker,
        buy_counter,
        sell_counter,
        good_buy_counter,
        bad_buy_counter,
        good_sell_counter,
        bad_sell_counter,
        hold_counter,
        good_hold_counter,
        bad_hold_counter,
        waiting_streak,
        good_hold_streak,
        bad_hold_streak,
        holds_with_no_shares_counter,
        good_holds_with_no_shares_counter,
        bad_holds_with_no_shares_counter,
        buy_price,
        min_price,
        max_price,
    ):
        description = (
            f"[+] {predicted_action} Transaction #{counter}: {short_desc}. <br>"
            f"Min Price: {min_price:.2f}. <br> "
            f"Close Price: {close_price:.2f}. <br> "
            f"Max Price: {max_price:.2f}. <br> "
            f"Reward: {reward}. <br> "
            f"Reward Tracker: {reward_tracker}. <br> "
            f"Shares: {shares_holding}. <br> "
            f"Available Amount: {available_amount}. <br> "
            f"Portfolio Value: {portfolio_value}. <br> "
            f"Buy Counters: {buy_counter}. <br> "
            f"&emsp; Good Buy Counters: {good_buy_counter}. <br> "
            f"&emsp; BAD Buy Counters: {bad_buy_counter}. <br> "
            f"Sell Counters: {sell_counter}. <br> "
            f"&emsp; GOOD SELL Counters: {good_sell_counter}. <br> "
            f"&emsp; BAD SELL Counters: {bad_sell_counter}. <br> "
            f"Hold Counters: {hold_counter}. <br> "
            f"&emsp; GOOD HOLD Counters: {good_hold_counter}. <br> "
            f"&emsp; BAD HOLD Counters: {bad_hold_counter}. <br> "
            f"Waiting Period Before Purchase: {waiting_streak} intervals. <br> "
            f"Good Streak: {good_hold_streak}. <br>"
            f"Bad Streak: {bad_hold_streak}. <br>"
            f"Holding with No Shares Counter: {holds_with_no_shares_counter} intervals. <br> "
            f"&emsp; GOOD HOLD with No Shares Counters: {good_holds_with_no_shares_counter}. <br> "
            f"&emsp; BAD HOLD with No Shares Counters: {bad_holds_with_no_shares_counter}. <br> "
            f"Buying Price: ₹{buy_price}. <br> "
        )
        return description

    def step(self, action):
        reward = 0
        terminated = False
        counter = self.state[0]
        past_n_prices = self.state[1:40]
        close_price = past_n_prices[-1]
        max_price = max(past_n_prices)
        min_price = min(past_n_prices)
        percentage_distance_from_min = self.state[41]
        percentage_distance_from_max = self.state[42]
        correct_trade = self.state[43]
        wrong_trade = self.state[44]
        buy_counter = self.state[45]
        sell_counter = self.state[46]
        hold_counter = self.state[47]
        good_buy_counter = self.state[48]
        good_sell_counter = self.state[49]
        good_hold_counter = self.state[50]
        bad_buy_counter = self.state[51]
        bad_sell_counter = self.state[52]
        bad_hold_counter = self.state[53]
        holds_with_no_shares_counter = self.state[54]
        good_holds_with_no_shares_counter = self.state[55]
        bad_holds_with_no_shares_counter = self.state[56]
        good_hold_profit = self.state[57]
        good_sell_profit = self.state[58]
        good_buy_profit = self.state[59]
        bad_buy_loss = self.state[60]
        bad_sell_loss = self.state[61]
        bad_hold_loss = self.state[62]
        good_profit = self.state[63]
        bad_loss = self.state[64]
        good_hold_streak = self.state[65]
        bad_hold_streak = self.state[66]
        waiting_streak = self.state[67]
        buy_price_index = self.state[68]
        reward_tracker = self.state[69]
        available_amount = self.state[70]
        shares_holding = self.state[71]
        buy_price = self.state[72]
        total_buy_price = self.state[73]
        profit = self.state[74]
        portfolio_value = self.state[75]
        portfolio_value_threshold = self.state[76]


        predicted_action = ACTION_MAP[action]


        # Attempting to buy without sufficient funds.
        if predicted_action == "BUY" and close_price > available_amount:
            reward -= 50_000
            terminated = True
            wrong_trade += 1
            short_desc = "BUY without funds"
            description = self.log(
                predicted_action,
                counter,
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                reward_tracker,
                buy_counter,
                sell_counter,
                good_buy_counter,
                bad_buy_counter,
                good_sell_counter,
                bad_sell_counter,
                hold_counter,
                good_hold_counter,
                bad_hold_counter,
                waiting_streak,
                good_hold_streak,
                bad_hold_streak,
                holds_with_no_shares_counter,
                good_holds_with_no_shares_counter,
                bad_holds_with_no_shares_counter,
                buy_price,
                min_price,
                max_price,
            )

        # Attempting to sell when no shares are held.
        elif predicted_action == "SELL" and shares_holding == 0:
            reward -= 50_000
            terminated = True
            wrong_trade += 1
            short_desc = "SELL without Shares"
            description = self.log(
                predicted_action,
                counter,
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                reward_tracker,
                buy_counter,
                sell_counter,
                good_buy_counter,
                bad_buy_counter,
                good_sell_counter,
                bad_sell_counter,
                hold_counter,
                good_hold_counter,
                bad_hold_counter,
                waiting_streak,
                good_hold_streak,
                bad_hold_streak,
                holds_with_no_shares_counter,
                good_holds_with_no_shares_counter,
                bad_holds_with_no_shares_counter,
                buy_price,
                min_price,
                max_price,
            )

        # Attempting to sell right after buying
        elif predicted_action == "SELL" and (
            good_hold_streak < 2 or bad_hold_streak < 2
        ):
            reward -= 50_000
            terminated = True
            wrong_trade += 1
            short_desc = "SELLING DIRECT AFTER BUY"
            description = self.log(
                predicted_action,
                counter,
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                reward_tracker,
                buy_counter,
                sell_counter,
                good_buy_counter,
                bad_buy_counter,
                good_sell_counter,
                bad_sell_counter,
                hold_counter,
                good_hold_counter,
                bad_hold_counter,
                waiting_streak,
                good_hold_streak,
                bad_hold_streak,
                holds_with_no_shares_counter,
                good_holds_with_no_shares_counter,
                bad_holds_with_no_shares_counter,
                buy_price,
                min_price,
                max_price,
            )

        # PF reached less than 75% of its initial price
        elif portfolio_value < 7500:
            reward -= 50_000
            terminated = True
            wrong_trade += 1
            short_desc = "PORTFOLIO VALUE is less than 7500"
            description = self.log(
                predicted_action,
                counter,
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                reward_tracker,
                buy_counter,
                sell_counter,
                good_buy_counter,
                bad_buy_counter,
                good_sell_counter,
                bad_sell_counter,
                hold_counter,
                good_hold_counter,
                bad_hold_counter,
                waiting_streak,
                good_hold_streak,
                bad_hold_streak,
                holds_with_no_shares_counter,
                good_holds_with_no_shares_counter,
                bad_holds_with_no_shares_counter,
                buy_price,
                min_price,
                max_price,
            )
        
        # Attempting to keeping missing good opportunities
        elif bad_holds_with_no_shares_counter > 200:
            reward -= 50_000
            terminated = True
            wrong_trade += 1
            short_desc = "missed 200+ opportunity"
            description = self.log(
                predicted_action,
                counter,
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                reward_tracker,
                buy_counter,
                sell_counter,
                good_buy_counter,
                bad_buy_counter,
                good_sell_counter,
                bad_sell_counter,
                hold_counter,
                good_hold_counter,
                bad_hold_counter,
                waiting_streak,
                good_hold_streak,
                bad_hold_streak,
                holds_with_no_shares_counter,
                good_holds_with_no_shares_counter,
                bad_holds_with_no_shares_counter,
                buy_price,
                min_price,
                max_price,
            )

        elif predicted_action == "BUY" and close_price <= available_amount:
            shares_bought = available_amount // close_price
            buy_price = close_price
            shares_holding += shares_bought
            total_buy_price = buy_price * shares_bought
            buy_price_index = counter
            available_amount -= total_buy_price

            if percentage_distance_from_min < 10:
                good_buy_counter += 1
                short_desc = "GOOD"
                reward += (
                    (max_price - min_price) * shares_bought
                )
            else:
                bad_buy_counter += 1
                short_desc = "BAD"
                reward -= (
                    (min_price - max_price) * shares_bought
                )

            short_desc = f"{short_desc} Purchase Successful"
            description = self.log(
                predicted_action,
                counter,
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                reward_tracker,
                buy_counter,
                sell_counter,
                good_buy_counter,
                bad_buy_counter,
                good_sell_counter,
                bad_sell_counter,
                hold_counter,
                good_hold_counter,
                bad_hold_counter,
                waiting_streak,
                good_hold_streak,
                bad_hold_streak,
                holds_with_no_shares_counter,
                good_holds_with_no_shares_counter,
                bad_holds_with_no_shares_counter,
                buy_price,
                min_price,
                max_price,
            )

            buy_counter += 1
            good_hold_streak = 0
            bad_hold_streak = 0
            waiting_streak = 0
            correct_trade += 1

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
                good_sell_counter += 1
                good_sell_profit += profit
                portfolio_value_threshold = portfolio_value
            else:
                short_desc = "UnProfitable"
                bad_sell_counter += 1
                bad_sell_loss += profit

            
            short_desc = f"{short_desc} Sale Executed."
            description = self.log(
                predicted_action,
                counter,
                short_desc,
                shares_holding,
                close_price,
                available_amount,
                portfolio_value,
                reward,
                reward_tracker,
                buy_counter,
                sell_counter,
                good_buy_counter,
                bad_buy_counter,
                good_sell_counter,
                bad_sell_counter,
                hold_counter,
                good_hold_counter,
                bad_hold_counter,
                waiting_streak,
                good_hold_streak,
                bad_hold_streak,
                holds_with_no_shares_counter,
                good_holds_with_no_shares_counter,
                bad_holds_with_no_shares_counter,
                buy_price,
                min_price,
                max_price,
            )

            sell_counter += 1
            shares_holding = 0
            buy_price = 0
            buy_price_index = -1
            total_buy_price = 0
            good_hold_streak = 0
            bad_hold_streak = 0


        elif predicted_action == "HOLD":
            if shares_holding == 0:

                if percentage_distance_from_min <= 10:
                    reward += (
                        min_price - max_price
                    ) * 10
                    short_desc = "MISSED Buying Opportunity"
                    bad_holds_with_no_shares_counter += 1
                else:
                    reward += (
                        max_price - min_price
                    ) * 10
                    short_desc = "Waiting for Good Opportunity"
                    good_holds_with_no_shares_counter += 1

                short_desc = f"{short_desc}"
                description = self.log(
                    predicted_action,
                    counter,
                    short_desc,
                    shares_holding,
                    close_price,
                    available_amount,
                    portfolio_value,
                    reward,
                    reward_tracker,
                    buy_counter,
                    sell_counter,
                    good_buy_counter,
                    bad_buy_counter,
                    good_sell_counter,
                    bad_sell_counter,
                    hold_counter,
                    good_hold_counter,
                    bad_hold_counter,
                    waiting_streak,
                    good_hold_streak,
                    bad_hold_streak,
                    holds_with_no_shares_counter,
                    good_holds_with_no_shares_counter,
                    bad_holds_with_no_shares_counter,
                    buy_price,
                    min_price,
                    max_price,
                )

                waiting_streak += 1
                holds_with_no_shares_counter += 1

            else:

                portfolio_value = shares_holding * close_price + available_amount
                profit = portfolio_value - portfolio_value_threshold
                reward += profit * shares_holding
                if portfolio_value > portfolio_value_threshold:
                    short_desc = "Profitable Holding"
                    good_hold_counter += 1
                    good_hold_profit += profit
                    good_hold_streak += 1
                    bad_hold_streak = 0
                else:
                    short_desc = "UnProfitable Holding"
                    bad_hold_counter += 1
                    bad_hold_loss += profit
                    bad_hold_streak += 1
                    good_hold_streak = 0

                short_desc = f"{short_desc}"
                description = self.log(
                    predicted_action,
                    counter,
                    short_desc,
                    shares_holding,
                    close_price,
                    available_amount,
                    portfolio_value,
                    reward,
                    reward_tracker,
                    buy_counter,
                    sell_counter,
                    good_buy_counter,
                    bad_buy_counter,
                    good_sell_counter,
                    bad_sell_counter,
                    hold_counter,
                    good_hold_counter,
                    bad_hold_counter,
                    waiting_streak,
                    good_hold_streak,
                    bad_hold_streak,
                    holds_with_no_shares_counter,
                    good_holds_with_no_shares_counter,
                    bad_holds_with_no_shares_counter,
                    buy_price,
                    min_price,
                    max_price,
                )

                hold_counter += 1
                waiting_streak = 0
        else:
            print(predicted_action)
            ValueError("Something is wrong in conditions")


        reward_tracker += reward
        done = counter == (self.length - 1)

        # correct_trade_percent = self.calculate_percent(
        #     correct_trade, correct_trade + wrong_trade
        # )
        # good_sell_counter_percent = self.calculate_percent(
        #     good_sell_counter, correct_trade + wrong_trade
        # )
        # good_hold_counter_percent = self.calculate_percent(
        #     good_hold_counter, correct_trade + wrong_trade
        # )

        # bad_sell_counter_percent = self.calculate_percent(
        #     bad_sell_counter, correct_trade + wrong_trade
        # )
        # bad_hold_counter_percent = self.calculate_percent(
        #     bad_hold_counter, correct_trade + wrong_trade
        # )
        # buy_counter_percent = self.calculate_percent(
        #     buy_counter, correct_trade + wrong_trade
        # )

        # holds_with_no_shares_counter_percent = self.calculate_percent(
        #     holds_with_no_shares_counter, correct_trade + wrong_trade
        # )


        combined_hold_profit = good_hold_profit + bad_hold_loss
        combined_sell_profit = good_sell_profit + bad_sell_loss

        info = {
            "seed": self.seed,
            "counter": counter,
            "min_price": min_price,
            "max_price": max_price,
            "close_price": close_price,
            "predicted_action": predicted_action,
            "description": description,
            "available_amount": available_amount,
            "shares_holdings": shares_holding,
            "buy_price": buy_price,
            "buy_price_index": buy_price_index,
            "reward": reward,
            "done": done,
            "terminated": terminated,
            "percentage_distance_from_min": percentage_distance_from_min,
            "percentage_distance_from_max": percentage_distance_from_max,
            "correct_trade": correct_trade,
            "wrong_trade": wrong_trade,
            # "correct_trade %": correct_trade_percent,
            "buy_counter": buy_counter,
            "sell_counter": sell_counter,
            "hold_counter": hold_counter,
            "good_hold_counter": good_hold_counter,
            "good_sell_counter": good_sell_counter,
            "good_buy_counter": good_buy_counter,
            "bad_hold_counter": bad_hold_counter,
            "bad_sell_counter": bad_sell_counter,
            "bad_buy_counter": bad_buy_counter,
            "hold_with_no_shares_counter": holds_with_no_shares_counter,
            "bad_holds_with_no_shares_counter": bad_holds_with_no_shares_counter,
            "good_holds_with_no_shares_counter": good_holds_with_no_shares_counter,
            "good_hold_streak": good_hold_streak,
            "bad_hold_streak": bad_hold_streak,
            "waiting_streak": waiting_streak,
            # "buy_counter %": buy_counter_percent,
            # "good_sell_counter %": good_sell_counter_percent,
            # "good_hold_counter %": good_hold_counter_percent,
            # "bad_sell_counter %": bad_sell_counter_percent,
            # "bad_hold_counter %": bad_hold_counter_percent,
            # "holds_with_no_shares_counter %": holds_with_no_shares_counter_percent,
            "good_hold_profit": good_hold_profit,
            "good_sell_profit": good_sell_profit,
            "good_buy_profit": good_buy_profit,
            "bad_hold_loss": bad_hold_loss,
            "bad_sell_loss": bad_sell_loss,
            "bad_buy_loss": bad_buy_loss,
            # "good_moves": good_moves,
            # "bad_moves": bad_moves,
            # "good_moves %": good_moves_percent,
            "reward_tracker": reward_tracker,
            "portfolio_value": portfolio_value,
            "portfolio_value_threshold": portfolio_value_threshold,
            "combined_hold_profit": combined_hold_profit,
            "combined_sell_profit": combined_sell_profit,
            "combined_total_profit": combined_hold_profit + combined_sell_profit,
            "combined_hold_streak": good_hold_streak + bad_hold_streak,
            # "track_portfolio": track_portfolio,
        }

        if done or terminated:
            return self.state, float(reward), bool(done), bool(terminated), info

        counter += 1
        past_n_prices = self.close_prices[int(counter)]
        close_price = past_n_prices[-1]
        min_price = min(past_n_prices)
        max_price = max(past_n_prices)
        average_price = sum(past_n_prices) / len(past_n_prices)
        percentage_distance_from_min = (
            (close_price - min_price) / (max_price - min_price)
        ) * 100
        percentage_distance_from_max = 100 - percentage_distance_from_min

        self.state = np.concatenate(
            (
                np.array([counter]),
                past_n_prices,
                np.array(
                    (
                        percentage_distance_from_min,
                        percentage_distance_from_max,
                        correct_trade,
                        wrong_trade,
                        buy_counter,
                        sell_counter,
                        hold_counter,
                        good_buy_counter,
                        good_sell_counter,
                        good_hold_counter,
                        bad_buy_counter,
                        bad_sell_counter,
                        bad_hold_counter,
                        holds_with_no_shares_counter,
                        good_holds_with_no_shares_counter,
                        bad_holds_with_no_shares_counter,
                        good_hold_profit,
                        good_sell_profit,
                        good_buy_profit,
                        bad_buy_loss,
                        bad_sell_loss,
                        bad_hold_loss,
                        good_profit,
                        bad_loss,
                        good_hold_streak,
                        bad_hold_streak,
                        waiting_streak,
                        buy_price_index,
                        reward_tracker
                    )
                ),
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
        return self.state, float(reward), bool(done), bool(terminated), info

    def close(self):
        pass

    def calculate_percent(self, v1, v2):
        if v1 == 0 or v2 == 0:
            return 0
        return round((v1 / (v2)) * 100, 2)
