import random
import json
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt



class ExpectVwapEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        super().__init__()

        # 초기 설정값
        self.MAX_NUM_SHARES = 1e6
        self.MAX_SHARE_PRICE = 1e6
        self.MAX_STEPS = len(df)

        # 데이터 프레임 형식: ['time', open', 'high', 'low', 'close', 'volume']
        self.df = df

        # 초기 설정값
        # shares_buy = 각 step에서 산 주식 수
        self.shares_buy = 0

        # # gym 라이브러리의 spaces 모듈을 사용하여 action_space를 설정
        # # 0~MAX NUM SHARES: Trading Volume

        # 0~MAX NUM SHARES: Trading Volume
        self.action_space = spaces.Box(
            low=np.array([0]), high=np.array([self.MAX_NUM_SHARES]), dtype=np.float32)

        # 현재까지 관찰된 주식 데이터를 관찰(시가, 종가, 고가, 저가, 거래량)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.MAX_STEPS, 6), dtype=np.float32)

        self.plot_data = []
        self.shares_held_data = []
        self.market_vwap_data = []

    def _next_observation(self):
        # 장시작부터 Current Step 이전까지의 데이터를 0~1 사이로 스케일링
        frame = np.array([
            self.df.loc[:self.current_step, 'Time'].values / self.MAX_STEPS,
            # self.df.loc[:self.current_step, 'Open'].values / self.MAX_SHARE_PRICE,
            # self.df.loc[:self.current_step, 'High'].values / self.MAX_SHARE_PRICE,
            # self.df.loc[:self.current_step, 'Low'].values / self.MAX_SHARE_PRICE,
            # self.df.loc[:self.current_step, 'Close'].values / self.MAX_SHARE_PRICE,
            self.df.loc[:self.current_step, 'Volume'].values / self.MAX_NUM_SHARES
        ]).T

        # Observation은 장 시작부터 current step까지의 데이터를 포함
        obs = np.zeros((self.MAX_STEPS, 6))
        obs[:self.current_step + 1, :] = frame
        return obs

    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        ## 매수 매도 관망 action type 선택해줄 때
        # action_type = action[0]
        # volume = action[1]

        volume = action[0]

        # 주식을 Action Space에서 정한 양만큼 사고 평균 가격을 업데이트
        self.shares_buy = int(volume)
        prev_cost = self.cost_basis * self.shares_held
        additional_cost = current_price * self.shares_buy

        # 평균 단가 계산 (평균단가 == VWAP)
        if self.shares_held + self.shares_buy > 0:
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + self.shares_buy)
        else:
            self.cost_basis = 0

        self.shares_held += self.shares_buy


    def step(self, action):
        # action을 취하고, reward를 계산
        self._take_action(action)
        self.current_step += 1

        # 종료 조건
        terminated = (self.current_step >= self.MAX_STEPS)
        if terminated:
            self.current_step = 0
            print("Episode terminated")
        #
        # # reward 계산
        # # 몇 주 이상은 가지고 있어야 한다
        # # Market VWAP - Our VWAP
        # market_vwap = ((self.df.loc[:self.current_step, 'Close'] * self.df.loc[:self.current_step, 'Volume']).values.sum()) / self.df.loc[:self.current_step, 'Volume'].values.sum()
        # remain_time = self.MAX_STEPS - self.current_step
        # reward = (market_vwap - self.cost_basis) / market_vwap * 100 / remain_time


        # market_volume = self.df.loc[self.current_step, 'Volume']
        # model_volume = self.shares_buy
        #
        # reward = -(abs(market_volume - model_volume))
        # reward = 0

        if self.cost_basis == 0:
            reward = 0

        # print("reward: ", reward)

        # truncated
        truncated = False

        # 다음 observation
        observation = self._next_observation()
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # 랜덤 seed 설정
        if seed is not None:
            ss = SeedSequence(seed)
            self.rng = default_rng(ss)
        else:
            self.rng = default_rng()

        # 상태 초기화
        self.shares_held = 0
        self.cost_basis = 0

        # Current Step을 0으로 설정
        self.current_step = 0

        return self._next_observation(), {}

    def render(self, mode='human', close=False):
        market_vwap = ((self.df.loc[:self.current_step, 'Close'] * self.df.loc[:self.current_step, 'Volume']).values.sum()) / self.df.loc[:self.current_step, 'Volume'].values.sum()
        # Render the environment to the screen
        vwap_gap = market_vwap - self.cost_basis

        state = {
            # 'Step': self.current_step,
            'Shares held': self.shares_held,
            'Model VWAP': self.cost_basis,
            'Market VWAP': market_vwap,
            'VWAP Gap': vwap_gap
        }

        if mode == 'human':
            print(f"------------Step: {self.current_step}-----------------")
            for key, value in state.items():
                print(f"{key}: {value}")
            print("--------------------------------")
            self.plot_data.append(vwap_gap)
            self.shares_held_data.append(self.shares_held)
            self.market_vwap_data.append(market_vwap)


        else:
            raise ValueError("Invalid render mode. Choose 'human' or 'system'.")

    # render using matplotlib
    def render_plot(self):
        # df = pd.DataFrame([[self.plot_data,self.shares_held_data]], columns=['vwap_gap, shares_held'])

        df = pd.DataFrame(self.plot_data, columns=['vwap_gap'])
        df['shares_held'] = self.shares_held_data
        df['market_vwap'] = self.market_vwap_data

        plt.plot(df['market_vwap'], c='g', label='Market VWAP')
        plt.legend(loc='upper left')
        plt.show()

        plt.plot(df['vwap_gap'].iloc[30:], c='g', label='VWAP gap')
        plt.legend(loc='upper left')
        plt.show()

        df['shares_chg'] = df['shares_held'].diff()
        plt.plot(df['shares_chg'], c='r', label='Shares Buy at each step')
        plt.legend(loc='upper left')
        plt.show()


#
# '''
# DQN
# '''
#
# class DQNExpectVwapEnv(gym.Env):
#     metadata = {'render.modes': ['human']}
#     def __init__(self, df):
#         super().__init__()
#
#         # 초기 설정값
#         self.MAX_NUM_SHARES = int(1e6)
#         self.MAX_SHARE_PRICE = 1e6
#         self.MAX_STEPS = len(df)
#
#         # 데이터 프레임 형식: ['time', open', 'high', 'low', 'close', 'volume']
#         self.df = df
#
#         # 초기 설정값
#         # shares_buy = 각 step에서 산 주식 수
#         self.shares_buy = 0
#
#         # # gym 라이브러리의 spaces 모듈을 사용하여 action_space를 설정
#         # # 0~MAX NUM SHARES: Trading Volume
#         # self.action_space = spaces.Box(
#         #     low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.int32)
#
#         # 0~MAX NUM SHARES: Trading Volume
#         self.action_space = spaces.Discrete(1000)
#
#
#
#         # 현재까지 관찰된 주식 데이터를 관찰(시가, 종가, 고가, 저가, 거래량)
#         self.observation_space = spaces.Box(
#             low=0, high=1, shape=(self.MAX_STEPS, 6), dtype=np.int32)
#
#         self.plot_data = []
#         self.shares_held_data = []
#         self.market_vwap_data = []
#
#     def _next_observation(self):
#         # 장시작부터 Current Step 이전까지의 데이터를 0~1 사이로 스케일링
#         frame = np.array([
#             self.df.loc[:self.current_step, 'Time'].values / self.MAX_STEPS,
#             self.df.loc[:self.current_step, 'Open'].values / self.MAX_SHARE_PRICE,
#             self.df.loc[:self.current_step, 'High'].values / self.MAX_SHARE_PRICE,
#             self.df.loc[:self.current_step, 'Low'].values / self.MAX_SHARE_PRICE,
#             self.df.loc[:self.current_step, 'Close'].values / self.MAX_SHARE_PRICE,
#             self.df.loc[:self.current_step, 'Volume'].values / self.MAX_NUM_SHARES
#         ]).T
#
#         # Observation은 장 시작부터 current step까지의 데이터를 포함
#         obs = np.zeros((self.MAX_STEPS, 6))
#         obs[:self.current_step + 1, :] = frame
#         return obs
#
#     def _take_action(self, action):
#         current_price = random.uniform(
#             self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
#
#         ## 매수 매도 관망 action type 선택해줄 때
#         # action_type = action[0]
#         # volume = action[1]
#
#         volume = action
#
#         # 주식을 Action Space에서 정한 양만큼 사고 평균 가격을 업데이트
#         self.shares_buy = int(volume)
#         prev_cost = self.cost_basis * self.shares_held
#         additional_cost = current_price * self.shares_buy
#
#         # 평균 단가 계산 (평균단가 == VWAP)
#         if self.shares_held + self.shares_buy > 0:
#             self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + self.shares_buy)
#         else:
#             self.cost_basis = 0
#
#         self.shares_held += self.shares_buy
#
#
#     def step(self, action):
#         # action을 취하고, reward를 계산
#         self._take_action(action)
#         self.current_step += 1
#
#         # 종료 조건
#         terminated = (self.current_step >= self.MAX_STEPS)
#         if terminated:
#             self.current_step = 0
#             print("Episode terminated")
#
#         # reward 계산
#         # 몇 주 이상은 가지고 있어야 한다
#         # Market VWAP - Our VWAP
#         market_vwap = ((self.df.loc[:self.current_step, 'Close'] * self.df.loc[:self.current_step, 'Volume']).values.sum()) / self.df.loc[:self.current_step, 'Volume'].values.sum()
#         remain_time = self.MAX_STEPS - self.current_step
#         reward = (market_vwap - self.cost_basis) / market_vwap * 100 / remain_time
#
#
#         if self.cost_basis == 0:
#             reward = 0
#
#         # print("reward: ", reward)
#
#         # truncated
#         truncated = False
#
#         # 다음 observation
#         observation = self._next_observation()
#         info = {}
#         return observation, reward, terminated, truncated, info
#
#     def reset(self, seed=None, options=None):
#         # 랜덤 seed 설정
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
#         # Current Step을 0으로 설정
#         self.current_step = 0
#
#         return self._next_observation(), {}
#
#     def render(self, mode='human', close=False):
#         market_vwap = ((self.df.loc[:self.current_step, 'Close'] * self.df.loc[:self.current_step, 'Volume']).values.sum()) / self.df.loc[:self.current_step, 'Volume'].values.sum()
#         # Render the environment to the screen
#         vwap_gap = market_vwap - self.cost_basis
#
#         state = {
#             # 'Step': self.current_step,
#             'Shares held': self.shares_held,
#             'Model VWAP': self.cost_basis,
#             'Market VWAP': market_vwap,
#             'VWAP Gap': vwap_gap
#         }
#
#         if mode == 'human':
#             print(f"------------Step: {self.current_step}-----------------")
#             for key, value in state.items():
#                 print(f"{key}: {value}")
#             print("--------------------------------")
#             self.plot_data.append(vwap_gap)
#             self.shares_held_data.append(self.shares_held)
#             self.market_vwap_data.append(market_vwap)
#
#
#         else:
#             raise ValueError("Invalid render mode. Choose 'human' or 'system'.")
#
#     # render using matplotlib
#     def render_plot(self):
#         # df = pd.DataFrame([[self.plot_data,self.shares_held_data]], columns=['vwap_gap, shares_held'])
#
#         df = pd.DataFrame(self.plot_data, columns=['vwap_gap'])
#         df['shares_held'] = self.shares_held_data
#         df['market_vwap'] = self.market_vwap_data
#
#         plt.plot(df['market_vwap'], c='g', label='Market VWAP')
#         plt.legend(loc='upper left')
#         plt.show()
#
#         plt.plot(df['vwap_gap'].iloc[30:], c='g', label='VWAP gap')
#         plt.legend(loc='upper left')
#         plt.show()
#
#         df['shares_chg'] = df['shares_held'].diff()
#         plt.plot(df['shares_chg'], c='r', label='Shares Buy at each step')
#         plt.legend(loc='upper left')
#         plt.show()
