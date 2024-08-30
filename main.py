import gym
import json
import datetime as dt

# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common.vec_env import DummyVecEnv
# from StockTradingEnv import StockTradingEnv

# from env.StockTradingEnv import StockTradingEnv
# from env.ExpectVwapEnv import ExpectVwapEnv
# from env.ExpectVwapEnv import DQNExpectVwapEnv
# from env.VwapEnvTest import ExpectVwapEnv

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

import talib as ta
from env.ExpectVolumeEnv import ExpectVolumeEnv
from env.ExpectVolumeEnvDiscrete import ExpectVolumeEnvDiscrete

import pandas as pd
from numpy.random import SeedSequence, default_rng

import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import CheckpointCallback

import datetime


'''
reference
https://github.com/notadamking/Stock-Trading-Environment
'''

'''
Data
2024-02-01 KOSPI Intraday Data
'''

# Set seed for reproducibility
ss = SeedSequence(12345)
rng = default_rng(ss)

# Load data
df = pd.read_csv("data/test/test.csv", encoding='cp949')
# df = pd.read_csv("data/raw/kospi_minutes/[지수KOSPI계열]일중 시세정보(1분)(주문번호-2499-1)_20211105.csv", encoding='cp949')
# df = pd.read_csv("data/raw/kospi_minutes/[지수KOSPI계열]일중 시세정보(1분)(주문번호-2499-1)_20220215.csv", encoding='cp949')
# df = pd.read_csv("data/raw/kospi_minutes/[지수KOSPI계열]일중 시세정보(1분)(주문번호-2499-1)_20231106.csv", encoding='cp949')
df = df[df['지수명']=='코스피']
# 마지막 2개 행 제거
df = df.iloc[:-2]

data_date = str(df['거래일자'].iloc[0])

df = df[['거래시각', '시가', '고가', '저가', '종가', '거래량']]
df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']

print(df)

df = df.astype(float)
df = df.reset_index(drop=False)


# # Create environment
env = ExpectVolumeEnv(df)


callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=1000, min_evals=100, verbose=1)

eval_callback = EvalCallback(
    env,
    eval_freq = len(df),
    callback_on_new_best=callback_on_best,
    # callback_after_eval=stop_train_callback,
    verbose=1,
    best_model_save_path="./logs/"
)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=100000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)


# Create model (PPO)
model = PPO("MlpPolicy",
            env,
            learning_rate=0.00025,
            batch_size=128,
            verbose=1,
            )


# Total timesteps / Number of steps per episode = Number of episodes
model.learn(total_timesteps=len(df)*1000)

# # Save model
# model.save(f"./logs/ppo_vwap_predict_{datetime.datetime.now().strftime('%Y%m%d')}.zip")
model.save(f"./logs/ppo_vwap_predict_test.zip")

# model.load("./logs/ppo_vwap_predict_240828.zip")
# model.load("./logs/ppo_vwap_predict_20240830.zip")

obs, empty = env.reset()

print("mean: ", df['Close'].mean())
plt.plot(df['Volume'], label=f'{data_date} Market Volume')
plt.show()

plt.plot(df['Close'], label=f'{data_date} Market Close')
plt.show()

# Render each environment separately
for _ in range(len(df)-1):
    action, _states = model.predict(obs)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

env.render_plot(data_date=data_date)