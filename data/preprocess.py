import glob
import os
import numpy as np
import pandas as pd


def preprocess_kospi_data():
    # get the file names
    file_names = glob.glob("/kospi_daily/*.csv")

    print(file_names)
    # read the files
    data = pd.concat([pd.read_csv(f, encoding='cp949') for f in file_names])
    # drop the duplicates
    data = data.drop_duplicates()
    # save the data
    data = data[data['지수명'] == '코스피']
    data = data.sort_values(by='거래일자')
    data = data[['거래일자', '지수업종코드', '지수명', '시가', '종가', '고가', '저가', '거래량', '거래대금']]
    data = data.rename(columns={'거래일자': 'Date', '지수업종코드': 'Code', '지수명': 'Name', '시가': 'Open', '종가': 'Close', '고가': 'High', '저가': 'Low', '거래량': 'Volume', '거래대금': 'Amount'})
    data = data.dropna()
    data = data.reset_index(drop=True)
    data.to_csv("/home/ydm/PycharmProjects/DQN-Trading/data/kospi_preprocessed/KOSPI.csv", index=False, encoding='cp949')

    return data

# data = preprocess_kospi_data()

# df = pd.read_csv("/home/ydm/PycharmProjects/DQN-Trading/data/kospi_preprocessed/kospi.csv", encoding='cp949')
# print(df)