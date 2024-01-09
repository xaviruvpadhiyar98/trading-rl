import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from common.load_close_prices import load_close_prices
from random import randint
from secrets import SystemRandom



ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}
EPS = 37.54
EXPECTED_GROWTH_RATE = 10
SCALE_FACTOR = 10
HOLD_FACTOR = 5
TRANSACTION_PENALTY = 1
RISK_ADJUSTMENT_FACTOR = 0.75  # Adjusts reward based on risk
DIVERSIFICATION_BONUS = 0.5    # Bonus for diversification
MAX_LOSS_THRESHOLD = 0.1       # Maximum allowable loss as a fraction of portfolio
PROFIT_SCALING_FACTOR = 0.02
LOSS_PENALTY_FACTOR = 0.08
HOLD_REWARD = 0.01
HOLD_PENALTY = -0.01
TRADE_PENALTY = 0.05  # Significantly increased penalty for each trade
HOLDING_REWARD_BASE = 0.001  # Base reward for holding shares
NO_SHARE_HOLDING_REWARD = 0.002  # Reward for holding with no shares
INITIAL_AMOUNT = 10_000
TICKER = "WHIRLPOOL.NS"
TICKER = "WIPRO.NS"
CLOSE_PRICES = load_close_prices(TICKER)


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
        r = SystemRandom()
        self.starting_random_number = r.randint(0, len(CLOSE_PRICES)-250)

        self.close_prices = CLOSE_PRICES[self.starting_random_number:self.starting_random_number+200]
        self.length = len(self.close_prices)

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(64,),
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
        holds_with_no_shares_counter = 0

        good_hold_profit = 0
        good_sell_profit = 0

        bad_sell_loss = 0
        bad_hold_loss = 0

        hold_streak = 0
        waiting_streak = 0

        reward_tracker = 0

        past_n_prices = self.close_prices[counter]
        close_price = past_n_prices[-1]

        min_price = min(past_n_prices)
        max_price = max(past_n_prices)

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
        self.updated_portfolio_value = 1
        self.buy_tracker = "["
        self.sell_tracker = "["


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
                        holds_with_no_shares_counter,
                        good_hold_profit,
                        good_sell_profit,
                        bad_sell_loss,
                        bad_hold_loss,
                        hold_streak,
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



    def step(self, action):
        reward = 0
        terminated = False
        counter = self.state[0]
        past_n_prices = self.state[1:41]
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
        holds_with_no_shares_counter = self.state[48]
        good_hold_profit = self.state[49]
        good_sell_profit = self.state[50]
        bad_sell_loss = self.state[51]
        bad_hold_loss = self.state[52]
        hold_streak = self.state[53]
        waiting_streak = self.state[54]
        buy_price_index = self.state[55]
        reward_tracker = self.state[56]
        available_amount = self.state[57]
        shares_holding = self.state[58]
        buy_price = self.state[59]
        total_buy_price = self.state[60]
        profit = self.state[61]
        portfolio_value = self.state[62]
        portfolio_value_threshold = self.state[63]


        predicted_action = ACTION_MAP[action]


        # Attempting to buy without sufficient funds.
        if predicted_action == "BUY" and close_price > available_amount:
            terminated = True
            short_desc = "BUY without funds"


        # Attempting to sell when no shares are held.
        elif predicted_action == "SELL" and shares_holding == 0:
            terminated = True
            short_desc = "SELL without Shares"

        elif predicted_action == "BUY" and close_price <= available_amount:
            shares_bought = available_amount // close_price
            buy_price = close_price
            shares_holding += shares_bought
            total_buy_price = buy_price * shares_bought
            buy_price_index = counter
            available_amount -= total_buy_price
            short_desc = f"Purchase Successful"
            reward = -0.01

            buy_counter += 1
            hold_streak = 0
            waiting_streak = 0
            correct_trade += 1
            self.buy_tracker += f"({counter},{buy_price}),"

        elif predicted_action == "SELL" and shares_holding > 0:
            shares_sold = shares_holding
            sell_price = close_price
            total_sell_price = sell_price * shares_sold
            available_amount += total_sell_price
            profit = total_sell_price - total_buy_price
            portfolio_value = available_amount

            if profit < 0:
                short_desc = "LOSS of profit when SELL"
                bad_sell_loss += profit
                reward = -1
                wrong_trade += 1

            else:

                if hold_streak < 3:
                    short_desc = "DIRECT SELL AFTER Buy with PROFIT"
                    bad_sell_loss += profit
                    reward = -1
                    wrong_trade += 1

                elif portfolio_value > portfolio_value_threshold:
                    short_desc = "exceeded PV SOLD"
                    good_sell_profit += profit
                    portfolio_value_threshold = portfolio_value
                    self.updated_portfolio_value += 1
                    reward = 0.99
                    correct_trade += 1

                else:
                    wrong_trade += 1
                    reward = 0.5
                    short_desc = "Profitable SOLD"
                    good_sell_profit += profit


            short_desc = f"{short_desc} Sale Executed."
            self.sell_tracker += f"({counter},{sell_price},{short_desc},{profit},{reward}),"

            sell_counter += 1
            shares_holding = 0
            buy_price = 0
            buy_price_index = -1
            total_buy_price = 0
            hold_streak = 0


        elif predicted_action == "HOLD":
            if shares_holding == 0:
                short_desc = f"Waiting for buying shares"
                waiting_streak += 1
                holds_with_no_shares_counter += 1


            else:
                portfolio_value = shares_holding * close_price + available_amount
                hold_streak += 1
                short_desc = "Holding with shares"
                hold_counter += 1
                waiting_streak = 0

                if portfolio_value > portfolio_value_threshold:
                    good_hold_profit = portfolio_value - portfolio_value_threshold
                    correct_trade += 1
                    reward = 0.6
                else:
                    wrong_trade += 1
                    bad_hold_loss = portfolio_value_threshold - portfolio_value
                    reward = -1

        else:
            print(predicted_action)
            ValueError("Something is wrong in conditions")



        # # Churning penalty
        # if (buy_counter+sell_counter) > 10:
        #     reward += -0.1 * (buy_counter+sell_counter - 10)

        base_value = 10000
        increment = 0.0001
        if portfolio_value > base_value:
            reward_increment = ((portfolio_value - base_value) // 100) * increment
            reward += reward_increment + 0.001

        if portfolio_value < portfolio_value_threshold and sell_counter > 1:
            reward -= 0.1
        
        # if portfolio_value > portfolio_value_threshold:
        #     reward += 0.002
        
        if portfolio_value == 10_000 and counter > 30:
            reward -= 1


        # reward = max(min(reward, 1), -1)

        # # Attempting to sell right after buying
        # if predicted_action == "SELL" and (
        #     hold_streak < 2
        # ):
        #     terminated = True
        #     short_desc = "SELLING DIRECT AFTER BUY"

        # if done nothing
        # elif buy_counter == 0 and sell_counter == 0 and counter > 20:
        #     terminated = True
        #     short_desc = "Didnt do anything"
            
        # # if done nothing
        # elif waiting_streak > 400:
        #     terminated = True
        #     short_desc = "keeps waiting"

        # PF reached less than 95% of its initial price
        if portfolio_value < 10_000:
            terminated = True
            short_desc = "PORTFOLIO VALUE is less than 10_000"
        

        
        # else:
        #     pass


        
        # # Attempting to keeping missing good opportunities
        # elif holds_with_no_shares_counter > 200:
        #     terminated = True
        #     short_desc = "missed 200+ opportunity"


        # Attempting to keeping missing good opportunities
        # elif bad_hold_counter > 10:
        #     terminated = True
        #     short_desc = "keeps holding badly"






        # if portfolio_value == portfolio_value_threshold:
        #     reward += 0
        # elif portfolio_value > portfolio_value_threshold:
        #     reward += (portfolio_value - portfolio_value_threshold) * self.updated_portfolio_value
        #     good_hold_profit = portfolio_value - portfolio_value_threshold
        # else:
        #     reward -= (portfolio_value_threshold - portfolio_value)
        #     bad_hold_loss = (portfolio_value_threshold - portfolio_value)



        # reward += portfolio_value - 10000

        # PF reached less than 95% of its initial price
        # if portfolio_value < 10_000:
        #     terminated = True
        #     short_desc = "PORTFOLIO VALUE is less than 10_000"


        if terminated:
            reward -= 2
            # wrong_trade += 1


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
            f"Portfolio Value threshold: {portfolio_value_threshold}. <br> "
            f"Buy Counters: {buy_counter}. <br> "
            f"Sell Counters: {sell_counter}. <br> "
            f"Hold Counters: {hold_counter}. <br> "
            f"Waiting Period Before Purchase: {waiting_streak} intervals. <br> "
            f"HOLD Streak: {hold_streak}. <br>"
            f"Holding with No Shares Counter: {holds_with_no_shares_counter} intervals. <br> "
            f"Buying Price: ₹{buy_price}. <br> "
            f"Buying Price Index: ₹{buy_price_index}. <br> "
            f"Portfolio Value counter: {self.updated_portfolio_value}. <br> "
            f"Buy Tracker: {self.buy_tracker}. <br>"
            f"Sell Tracker: {self.sell_tracker}. <br>"
        )


        reward_tracker += reward
        done = counter == (self.length - 1)




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
            "buy_counter": buy_counter,
            "sell_counter": sell_counter,
            "hold_counter": hold_counter,
            "hold_with_no_shares_counter": holds_with_no_shares_counter,
            "hold_streak": hold_streak,
            "waiting_streak": waiting_streak,
            "good_hold_profit": good_hold_profit,
            "good_sell_profit": good_sell_profit,
            "bad_hold_loss": bad_hold_loss,
            "bad_sell_loss": bad_sell_loss,
            "reward_tracker": reward_tracker,
            "portfolio_value": portfolio_value,
            "portfolio_value_threshold": portfolio_value_threshold,
            "combined_hold_profit": combined_hold_profit,
            "combined_sell_profit": combined_sell_profit,
            "combined_total_profit": combined_hold_profit + combined_sell_profit,
            "updated_portfolio_value": self.updated_portfolio_value,
            "buy_tracker": self.buy_tracker,
            "sell_tracker": self.sell_tracker,
            "starting_random_number": self.starting_random_number
        }

        if done or terminated:
            self.buy_tracker += "]"
            self.sell_tracker += "]"
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
                        holds_with_no_shares_counter,
                        good_hold_profit,
                        good_sell_profit,
                        bad_sell_loss,
                        bad_hold_loss,
                        hold_streak,
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
