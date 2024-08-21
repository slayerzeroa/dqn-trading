import pandas as pd

pd.set_option('display.max_rows', None)

df = pd.read_csv("kospi_daily/[지수KOSPI계열]일별 시세정보(주문번호-2162-3)_199607.csv", encoding='cp949')

print(df)