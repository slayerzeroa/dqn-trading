import json
import datetime

import pandas as pd
import numpy as np
from numpy.random import SeedSequence, default_rng
import random

import gym
import talib as ta

import torch

from env.ExpectVolumeEnv import ExpectVolumeEnv
from env.ExpectVolumeEnvDiscrete import ExpectVolumeEnvDiscrete

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

import matplotlib.pyplot as plt


'''
reference
https://github.com/notadamking/Stock-Trading-Environment
'''

'''
Data
20XX-XX-XX KOSPI Intraday Data
'''

random_seed = 42

# Set seed for reproducibility
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


# Load data
df = pd.read_csv("data/raw/kospi_minutes/[지수KOSPI계열]일중 시세정보(1분)(주문번호-2499-1)_20240111.csv", encoding='cp949')

# DataFrame Preprocessing
df = df[df['지수명']=='코스피']
df = df[df['거래시각'] <= '1530']

data_date = str(df['거래일자'].iloc[0])

df = df[['거래시각', '시가', '고가', '저가', '종가', '거래량']]
df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']

print(df)

df = df.astype(float)
df = df.reset_index(drop=False)


# # Create environment
env = ExpectVolumeEnv(df)

# Create model (PPO)
model = PPO("MlpPolicy",
            env,
            learning_rate=0.00025,
            batch_size=128,
            verbose=1,
            )


# # Total timesteps / Number of steps per episode = Number of episodes
# model.learn(total_timesteps=len(df)*100)

# # # Save model
# model.save(f"./logs/ppo_vwap_predict_{datetime.datetime.now().strftime('%Y%m%d')}_{data_date}.zip")


# Load Model
model.load("./logs/ppo_vwap_predict_20240919_20240111.zip", env=env)

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

market_vwap = env.render_plot(data_date=data_date)

volume_pattern = pd.read_csv('./data/volume.csv')
scaled_mean = volume_pattern['scaled_mean']

proportion = scaled_mean / np.sum(scaled_mean)

static_model_vwap = np.sum(df['Close'] * proportion)
print(f"Static Model VWAP: {static_model_vwap}")
print(f"Static Model VWAP Gap: {market_vwap - static_model_vwap}")




# # Callbacks
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
# stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=1000, min_evals=100, verbose=1)
#
# eval_callback = EvalCallback(
#     env,
#     eval_freq = len(df),
#     callback_on_new_best=callback_on_best,
#     # callback_after_eval=stop_train_callback,
#     verbose=1,
#     best_model_save_path="./logs/"
# )
#
# # Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(
#   save_freq=100000,
#   save_path="./logs/",
#   name_prefix="rl_model",
#   save_replay_buffer=True,
#   save_vecnormalize=True,
# )