import gym
import json
import datetime as dt

# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common.vec_env import DummyVecEnv
# from StockTradingEnv import StockTradingEnv
from stable_baselines3 import PPO
import talib as ta
# from env.StockTradingEnv import StockTradingEnv
from env.ExpectVwapEnv import ExpectVwapEnv

import pandas as pd
from numpy.random import SeedSequence, default_rng

ss = SeedSequence(12345)
rng = default_rng(ss)

# Load data
df = pd.read_csv("data/coin_data/btc_result.csv", encoding='cp949')
print(df)

df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

env = ExpectVwapEnv(df)

model = PPO("MlpPolicy", env, verbose=1)

# Total timesteps is the number of steps that the agent takes in the environment
# Total timesteps / number of steps per episode = number of episodes
model.learn(total_timesteps=20000)

obs = env.reset()[0]

# Render each environment separately
for _ in range(20000):
    action, _states = model.predict(obs)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

env.render_plot()

# model.save("ppo2_vwap_predict")