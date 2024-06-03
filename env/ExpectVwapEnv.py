# import random
# import json
# import gymnasium as gym
# from gymnasium import spaces
# import pandas as pd
# import numpy as np
# from numpy.random import SeedSequence, default_rng
# import matplotlib.pyplot as plt
#
# # 초기 설정값
# MAX_AVERGAE_PRICE = 1000000000
# MAX_NUM_SHARES = 1000000000
# MAX_SHARE_PRICE = 1000000000
# MAX_STEPS = 300
#
# class ExpectVwapEnv(gym.Env):
#     metadata = {'render.modes': ['human']}
#     def __init__(self, df):
#         super().__init__()
#
#         # 데이터 프레임 형식: ['time', open', 'high', 'low', 'close', 'volume']
#         self.df = df
#         self.reward_range = (0, MAX_AVERGAE_PRICE)
#
#         # gym 라이브러리의 spaces 모듈을 사용하여 action_space를 설정
#         # 0~1:  Do nothing, Action
#         # 0~MAX NUM SHARES: Trading Volume
#         self.action_space = spaces.Box(
#             low=np.array([0, 0]), high=np.array([1, MAX_NUM_SHARES]), dtype=np.float16)
#
#         # 현재까지 관찰된 주식 데이터를 관찰(시가, 종가, 고가, 저가, 거래량)
#         self.observation_space = spaces.Box(
#             low=0, high=1, shape=(MAX_STEPS, 5), dtype=np.float16)
#
#         self.plot_data = []
#
#     def _next_observation(self):
#         # 5일 전까지의 주식 데이터를 가져와 0~1 사이로 스케일링
#         frame = np.array([
#             self.df.loc[:self.current_step, 'Open'].values / MAX_SHARE_PRICE,
#             self.df.loc[:self.current_step, 'High'].values / MAX_SHARE_PRICE,
#             self.df.loc[:self.current_step, 'Low'].values / MAX_SHARE_PRICE,
#             self.df.loc[:self.current_step, 'Close'].values / MAX_SHARE_PRICE,
#             self.df.loc[:self.current_step, 'Volume'].values / MAX_NUM_SHARES
#         ])
#         # print("before append shape: ", frame.shape)
#
#         # # 추가 데이터를 추가하고 각 값을 0-1 사이로 스케일링
#         # # balance: 현재 계좌 잔액(현금)
#         # # net worth: 현재 순자산
#         # # shares_held: 보유 주식 수
#         # # cost_basis: 보유 주식의 평균 가격
#         # # total_shares_sold: 총 판매 주식 수
#         # # total_sales_value: 총 판매 가치
#         # obs = np.append(frame, [[
#         #     self.balance / MAX_ACCOUNT_BALANCE,
#         #     self.max_net_worth / MAX_ACCOUNT_BALANCE,
#         #     self.shares_held / MAX_NUM_SHARES,
#         #     self.cost_basis / MAX_SHARE_PRICE,
#         #     self.total_shares_sold / MAX_NUM_SHARES,
#         #     self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
#         # ]], axis=0)
#
#         obs = frame
#
#         return obs
#
#     def _take_action(self, action):
#         # 현재 가격을 Time Step 내의 랜덤 가격으로 설정
#         current_price = random.uniform(
#             self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
#
#         # action[0]은 Action, Do nothing
#         # action[1]은 Number of Trading Volume
#         action_type = action[0]
#         volume = action[1]
#
#         if action_type < 0:
#             # Buy amount % of balance in shares
#             shares_bought = int(volume)
#             prev_cost = self.cost_basis * self.shares_held
#             additional_cost = shares_bought * current_price
#
#             self.cost_basis = (
#                 prev_cost + additional_cost) / (self.shares_held + shares_bought)
#             self.shares_held += shares_bought
#
#
#     def step(self, action):
#         # action을 취하고, reward를 계산
#         self._take_action(action)
#         self.current_step += 1
#
#         # 종료 조건
#         if self.current_step > MAX_STEPS:
#             self.current_step = 0
#
#         # reward 계산
#         # Market VWAP - Our VWAP
#         market_vwap = ((self.df.loc[:self.current_step, 'Close'] * self.df.loc[:self.current_step, 'Volume']).values.sum()) / self.df.loc[:self.current_step, 'Volume'].values.sum()
#         reward = market_vwap - self.cost_basis
#
#         terminated = self.current_step > MAX_STEPS
#         # truncated
#         truncated = False
#         # 다음 observation
#         observation = self._next_observation()
#         info = {}
#         return observation, reward, terminated, truncated, info
#
#     def reset(self, seed=None, options=None):
#         if seed is not None:
#             ss = SeedSequence(seed)
#             self.rng = default_rng(ss)
#         else:
#             self.rng = default_rng()
#
#         # 상태 초기화
#         self.shares_held = 0
#         self.cost_basis = 0
#
#         # Set the current step to a random point within the data frame
#         # 현재 가격을 Time Step 내의 랜덤 가격으로 설정
#         # self.current_step = random.randint(
#         #     0, len(self.df.loc[:, 'Open'].values) - 6)
#         self.current_step = self.rng.integers(0, len(self.df.loc[:, 'Open'].values))
#
#         return self._next_observation(), {}
#
#     def render(self, mode='human', close=False):
#         mv = ((self.df.loc[:MAX_STEPS, 'Close'] * self.df.loc[:MAX_STEPS, 'Volume']).values.sum()) / self.df.loc[:MAX_STEPS, 'Volume'].values.sum()
#         # Render the environment to the screen
#         vwap_gap = mv - self.cost_basis
#
#         state = {
#             # 'Step': self.current_step,
#             'Shares held': self.shares_held,
#             'Model VWAP': self.cost_basis,
#             'Market VWAP': mv,
#             'VWAP Gap': vwap_gap
#         }
#
#
#         if mode == 'human':
#             print(f"------------Step: {self.current_step}-----------------")
#             for key, value in state.items():
#                 print(f"{key}: {value}")
#             print("--------------------------------")
#             self.plot_data.append([vwap_gap])
#
#         else:
#             raise ValueError("Invalid render mode. Choose 'human' or 'system'.")
#
#     # render using matplotlib
#
#     def render_plot(self):
#         df = pd.DataFrame(self.plot_data, columns=['vwap_gap'])
#         # plt.plot(df['balance'], c='r', label='Balance')
#         plt.plot(df['vwap_gap'], c='g', label='VWAP gap')
#         plt.legend(loc='upper left')
#         plt.show()


import random
import json
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt

# 초기 설정값
MAX_AVERGAE_PRICE = 1e6
MAX_NUM_SHARES = 1e6
MAX_SHARE_PRICE = 1e6
MAX_STEPS = 1400

class ExpectVwapEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        super().__init__()

        # 데이터 프레임 형식: ['time', open', 'high', 'low', 'close', 'volume']
        self.df = df
        self.reward_range = (0, MAX_AVERGAE_PRICE)


        # gym 라이브러리의 spaces 모듈을 사용하여 action_space를 설정
        # 0~1:  Do nothing, Action
        # 0~MAX NUM SHARES: Trading Volume
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([1, MAX_NUM_SHARES]), dtype=np.float32)

        # 현재까지 관찰된 주식 데이터를 관찰(시가, 종가, 고가, 저가, 거래량)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(MAX_STEPS, 5), dtype=np.float32)

        self.plot_data = []

    def _next_observation(self):
        # 5일 전까지의 주식 데이터를 가져와 0~1 사이로 스케일링
        frame = np.array([
            self.df.loc[:self.current_step, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[:self.current_step, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[:self.current_step, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[:self.current_step, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[:self.current_step, 'Volume'].values / MAX_NUM_SHARES
        ]).T
        # print(self.current_step)

        obs = np.zeros((MAX_STEPS, 5))
        obs[-frame.shape[0]:] = frame
        return obs

    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        volume = action[1]

        if action_type < 1:
            shares_bought = int(volume)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            if self.shares_held + shares_bought > 0:
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            else:
                self.cost_basis = 0

            self.shares_held += shares_bought


    def step(self, action):
        # action을 취하고, reward를 계산
        self._take_action(action)
        self.current_step += 1

        # 종료 조건
        if self.current_step >= MAX_STEPS:
            self.current_step = 0

        # reward 계산
        # Market VWAP - Our VWAP
        market_vwap = ((self.df.loc[:self.current_step, 'Close'] * self.df.loc[:self.current_step, 'Volume']).values.sum()) / self.df.loc[:self.current_step, 'Volume'].values.sum()
        reward = market_vwap - self.cost_basis

        terminated = self.current_step >= MAX_STEPS
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
        self.shares_held = 0
        self.cost_basis = 0

        # Set the current step to a random point within the data frame
        # 현재 가격을 Time Step 내의 랜덤 가격으로 설정
        self.current_step = self.rng.integers(0, len(self.df.loc[:, 'Open'].values))

        return self._next_observation(), {}

    def render(self, mode='human', close=False):
        mv = ((self.df.loc[:MAX_STEPS, 'Close'] * self.df.loc[:MAX_STEPS, 'Volume']).values.sum()) / self.df.loc[:MAX_STEPS, 'Volume'].values.sum()
        # Render the environment to the screen
        vwap_gap = mv - self.cost_basis

        state = {
            # 'Step': self.current_step,
            'Shares held': self.shares_held,
            'Model VWAP': self.cost_basis,
            'Market VWAP': mv,
            'VWAP Gap': vwap_gap
        }

        if mode == 'human':
            print(f"------------Step: {self.current_step}-----------------")
            for key, value in state.items():
                print(f"{key}: {value}")
            print("--------------------------------")
            self.plot_data.append([vwap_gap])

        else:
            raise ValueError("Invalid render mode. Choose 'human' or 'system'.")

    # render using matplotlib
    def render_plot(self):
        df = pd.DataFrame(self.plot_data, columns=['vwap_gap'])
        # plt.plot(df['balance'], c='r', label='Balance')
        plt.plot(df['vwap_gap'].iloc[10:], c='g', label='VWAP gap')
        plt.legend(loc='upper left')
        plt.show()
