import gym
import json
import datetime as dt

# from stable_baslines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import talib as ta
from env.StockTradingEnv import StockTradingEnv

import pandas as pd

# Load data
df = pd.read_csv("data/kospi_preprocessed/KOSPI.csv", encoding='cp949')
df.dropna(inplace=True)
df = df.sort_values('Date')
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
print(df)


env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='human')

model.save("ppo2_stock_trading")