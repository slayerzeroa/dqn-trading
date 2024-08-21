# import gym
# import json
# import datetime as dt
#
# # from stable_baselines3.common.policies import MlpPolicy
# # from stable_baselines3.common.vec_env import DummyVecEnv
# # from StockTradingEnv import StockTradingEnv
#
# # from env.StockTradingEnv import StockTradingEnv
# # from env.ExpectVwapEnv import ExpectVwapEnv
# # from env.ExpectVwapEnv import DQNExpectVwapEnv
# # from env.VwapEnvTest import ExpectVwapEnv
#
# from stable_baselines3 import PPO
# from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
# from stable_baselines3.common.monitor import Monitor
# import talib as ta
# from env.ExpectVolumeEnv import ExpectVolumeEnv
#
# import pandas as pd
# from numpy.random import SeedSequence, default_rng
#
# import matplotlib.pyplot as plt
#
# from stable_baselines3.common.callbacks import CheckpointCallback
#
#
# '''
# reference
# https://github.com/notadamking/Stock-Trading-Environment
# '''
#
# # Set seed for reproducibility
# ss = SeedSequence(12345)
# rng = default_rng(ss)
#
# # Load data
# df = pd.read_csv("data/test/test.csv", encoding='cp949')
# df = df[df['지수명']=='코스피']
# # 마지막 2개 행 제거
# df = df.iloc[:-2]
#
# df = df[['거래시각', '시가', '고가', '저가', '종가', '거래량']]
# df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
#
# print(df)
#
# df = df.astype(float)
# df = df.reset_index(drop=False)
#
#
# # # Create environment
# env = ExpectVolumeEnv(df)
# # env = DQNExpectVwapEnv(df)
# # print(env)
#
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
# stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=1000, verbose=1)
#
# eval_callback = EvalCallback(
#     env,
#     eval_freq = len(df),
#     callback_on_new_best=callback_on_best,
#     callback_after_eval=stop_train_callback,
#     verbose=1,
#     best_model_save_path="./logs/"
# )
# #
# # # Save a checkpoint every 1000 steps
# # checkpoint_callback = CheckpointCallback(
# #   save_freq=1000000,
# #   save_path="./logs/",
# #   name_prefix="rl_model",
# #   save_replay_buffer=True,
# #   save_vecnormalize=True,
# # )
#
#
# # Create model
# model = PPO("MlpPolicy", env, learning_rate=0.001, batch_size=128, verbose=1)
#
#
# # Total timesteps / Number of steps per episode = Number of episodes
# model.learn(total_timesteps=len(df)*1000, callback=eval_callback)
#
# # Save model
# model.save("ppo2_vwap_predict")
#
# # model.load("./logs/best_model.zip")
#
#
# obs, empty = env.reset()
#
# print("mean: ", df['Close'].mean())
# plt.plot(df['Volume'])
# plt.show()
#
# plt.plot(df['Close'])
# plt.show()
#
# # Render each environment separately
# for _ in range(len(df)-1):
#     action, _states = model.predict(obs)
#     observation, reward, terminated, truncated, info = env.step(action)
#     env.render()
#
# env.render_plot()
#
#
#
#


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
import talib as ta
from env.ExpectVolumeEnv import ExpectVolumeEnv

import pandas as pd
from numpy.random import SeedSequence, default_rng

import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import CheckpointCallback


'''
reference
https://github.com/notadamking/Stock-Trading-Environment
'''

# Set seed for reproducibility
ss = SeedSequence(12345)
rng = default_rng(ss)

# Load data
df = pd.read_csv("data/test/test.csv", encoding='cp949')
df = df[df['지수명']=='코스피']
# 마지막 2개 행 제거
df = df.iloc[:-2]

df = df[['거래시각', '시가', '고가', '저가', '종가', '거래량']]
df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']

print(df)

df = df.astype(float)
df = df.reset_index(drop=False)


# # Create environment
env = ExpectVolumeEnv(df)
# env = DQNExpectVwapEnv(df)
# print(env)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)


# Create model
model = PPO("MlpPolicy", env, learning_rate=0.00025, batch_size=128, verbose=1)


# Total timesteps / Number of steps per episode = Number of episodes
model.learn(total_timesteps=len(df)*1000, callback=checkpoint_callback)

# Save model
model.save("ppo2_vwap_predict")


obs, empty = env.reset()

print("mean: ", df['Close'].mean())
plt.plot(df['Volume'])
plt.show()

plt.plot(df['Close'])
plt.show()

# Render each environment separately
for _ in range(len(df)-1):
    action, _states = model.predict(obs)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

env.render_plot()