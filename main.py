import gym
import json
import datetime as dt

# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common.vec_env import DummyVecEnv
# from StockTradingEnv import StockTradingEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
import talib as ta
# from env.StockTradingEnv import StockTradingEnv
from env.ExpectVwapEnv import ExpectVwapEnv

import pandas as pd
from numpy.random import SeedSequence, default_rng

import matplotlib.pyplot as plt

'''
reference: https://github.com/notadamking/Stock-Trading-Environment
'''

# Set seed for reproducibility
ss = SeedSequence(12345)
rng = default_rng(ss)

# Load data
df = pd.read_csv("data/coin_data/btc_result.csv", encoding='cp949')
df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


# Create environment
env = ExpectVwapEnv(df)

# Create model
model = PPO("MlpPolicy", env, verbose=1)

# Total timesteps / Number of steps per episode = Number of episodes
model.learn(total_timesteps=len(df)*10)

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



# model.save("ppo2_vwap_predict")