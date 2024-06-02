import random
import json
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt

# 초기 설정값
MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000 # 초기 투자금

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        super().__init__()

        # 데이터 프레임 형식: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # gym 라이브러리의 spaces 모듈을 사용하여 action_space를 설정
        # 0~3: Buy, Sell, Hold
        # 0~1: % of balance
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # 5일간의 주식 데이터와 5개의 데이터 포인트를 관찰(시가, 종가, 고가, 저가, 거래량)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

        self.plot_data = []

    def _next_observation(self):
        # 5일 전까지의 주식 데이터를 가져와 0~1 사이로 스케일링
        frame = np.array([
            self.df.loc[self.current_step : self.current_step + 5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step : self.current_step + 5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step : self.current_step + 5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step : self.current_step + 5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step : self.current_step + 5, 'Volume'].values / MAX_NUM_SHARES
        ])
        # print("before append shape: ", frame.shape)

        # 추가 데이터를 추가하고 각 값을 0-1 사이로 스케일링
        # balance: 현재 계좌 잔액(현금)
        # net worth: 현재 순자산
        # shares_held: 보유 주식 수
        # cost_basis: 보유 주식의 평균 가격
        # total_shares_sold: 총 판매 주식 수
        # total_sales_value: 총 판매 가치
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def _take_action(self, action):
        # 현재 가격을 Time Step 내의 랜덤 가격으로 설정
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        # action[0]은 Buy, Sell, Hold
        # action[1]은 % of balance
        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # 주식을 모두 팔았을 때 cost_basis를 0으로 설정
        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # action을 취하고, reward를 계산
        self._take_action(action)
        self.current_step += 1

        # 종료 조건
        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        # reward 계산
        delay_modifier = (self.current_step / MAX_STEPS)
        reward = self.balance * delay_modifier

        # terminated은 net worth가 0보다 작거나 같을 때
        terminated = self.net_worth <= 0
        # truncated
        truncated = False
        # 다음 observation
        observation = self._next_observation()
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            ss = SeedSequence(seed)
            self.rng = default_rng(ss)
        else:
            self.rng = default_rng()

        # 상태 초기화
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        # 현재 가격을 Time Step 내의 랜덤 가격으로 설정
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'Open'].values) - 6)
        self.current_step = self.rng.integers(0, len(self.df.loc[:, 'Open'].values) - 6)

        return self._next_observation(), {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        state = {
            # 'Step': self.current_step,
            'Balance': self.balance,
            'Shares held': self.shares_held,
            'Total sold': self.total_shares_sold,
            'Avg cost for held shares': self.cost_basis,
            'Total sales value': self.total_sales_value,
            'Net worth': self.net_worth,
            'Max net worth': self.max_net_worth,
            'Profit': profit
        }


        if mode == 'human':
            print(f"------------Step: {self.current_step}-----------------")
            for key, value in state.items():
                print(f"{key}: {value}")
            print("--------------------------------")
            self.plot_data.append([self.net_worth])

        else:
            raise ValueError("Invalid render mode. Choose 'human' or 'system'.")

    # render using matplotlib

    def render_plot(self):
        df = pd.DataFrame(self.plot_data, columns=['net_worth'])
        # plt.plot(df['balance'], c='r', label='Balance')
        plt.plot(df['net_worth'], c='g', label='Net Worth')
        plt.legend(loc='upper left')
        plt.show()
