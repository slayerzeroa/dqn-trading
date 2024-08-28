import random
import json
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt

class ExpectVolumeEnvDiscrete(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        super().__init__()

        # 초기 설정값
        self.MAX_NUM_SHARES = int(1e6)
        self.MAX_STEPS = len(df)

        # 데이터 프레임 형식: ['time', 'volume']
        self.df = df

        # 초기 설정값
        # shares_buy = 각 step에서 산 주식 수
        self.shares_buy = 0

        # # gym 라이브러리의 spaces 모듈을 사용하여 action_space를 설정
        # 0~MAX NUM SHARES: Trading Volume
        self.action_space = spaces.Discrete(self.MAX_NUM_SHARES)

        # observation_space를 Dict로 설정하고 goal 관련 키를 추가
        # observation_space를 단순화하여 중첩된 Dict를 피함
        self.observation_space = spaces.Dict({
            'time': spaces.Box(low=0, high=1, shape=(self.MAX_STEPS, 1), dtype=np.float32),
            'volume': spaces.Box(low=0, high=self.MAX_NUM_SHARES, shape=(self.MAX_STEPS, 1), dtype=np.float32),
            'achieved_goal': spaces.Box(low=0, high=self.MAX_NUM_SHARES, shape=(1,), dtype=np.float32),
            'desired_goal': spaces.Box(low=0, high=self.MAX_NUM_SHARES, shape=(1,), dtype=np.float32),
        })

        self.plot_data = []
        self.shares_held_data = []
        self.market_vwap_data = []

    def _next_observation(self):
        # 장시작부터 Current Step 이전까지의 데이터를 0~1 사이로 스케일링
        time_frame = self.df.loc[:self.current_step, 'Time'].values / self.MAX_STEPS
        volume_frame = self.df.loc[:self.current_step, 'Volume'].values / self.MAX_NUM_SHARES

        # time과 volume을 Dict로 반환
        observation = {
            'time': np.zeros((self.MAX_STEPS, 1)),
            'volume': np.zeros((self.MAX_STEPS, 1)),
            'achieved_goal': np.array([self.shares_buy], dtype=np.float32),
            'desired_goal': np.array([self.df.loc[self.current_step, 'Volume']], dtype=np.float32)
        }
        observation['time'][:self.current_step + 1, 0] = time_frame
        observation['volume'][:self.current_step + 1, 0] = volume_frame

        return observation

    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        # action은 이미 정수형 스칼라 값이므로, 바로 사용하면 됩니다.
        volume = action

        # 주식을 Action Space에서 정한 양만큼 사고 평균 가격을 업데이트
        self.shares_buy = volume
        prev_cost = self.cost_basis * self.shares_held
        additional_cost = current_price * self.shares_buy

        # 평균 단가 계산 (평균단가 == VWAP)
        if self.shares_held + self.shares_buy > 0:
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + self.shares_buy)
        else:
            self.cost_basis = 0

        self.shares_held += self.shares_buy

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.any(achieved_goal < 10):
            if np.any(achieved_goal < 0):
                reward = -100
            else:
                reward = -1
        if np.any(achieved_goal >= 10):
            reward =-(((achieved_goal - desired_goal) / self.MAX_NUM_SHARES) ** 2)
        # 목표와 달성된 목표를 비교하여 보상을 계산
        return reward

    def _calculate_reward(self, achieved_goal, desired_goal):
        if np.any(achieved_goal < 10):
            if np.any(achieved_goal < 0):
                reward = -100
            else:
                reward = -1
        if np.any(achieved_goal >= 10):
            reward =-(((achieved_goal - desired_goal) / self.MAX_NUM_SHARES) ** 2)
        # 목표와 달성된 목표를 비교하여 보상을 계산
        return reward


    def step(self, action):
        # action을 취하고, reward를 계산
        self._take_action(action)
        self.current_step += 1

        # 종료 조건
        terminated = (self.current_step >= self.MAX_STEPS)
        if terminated:
            self.current_step = 0
            print("Episode terminated")

        # # reward 계산
        # reward = -(((self.shares_buy - self.df.loc[self.current_step, 'Volume']) / self.MAX_NUM_SHARES) ** 2)

        if self.shares_buy < 10:
            if self.shares_buy < 0:
                reward = -100
            else:
                reward = -1
        if self.shares_buy >= 10:
            reward =-(((self.shares_buy - self.df.loc[self.current_step, 'Volume']) / self.MAX_NUM_SHARES) ** 2)

        # reward = -((self.shares_buy - self.df.loc[self.current_step, 'Volume'])**2)
        #
        # if self.cost_basis == 0:
        #     reward = 0

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