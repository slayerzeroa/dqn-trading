import pandas as pd
import datetime

from sklearn.preprocessing import minmax_scale

# # # Load data
# # day = '20000104'
#
times = []
time = '0900'
while time != '1531':
    times.append(time)
    time = datetime.datetime.strptime(time, '%H%M')
    time += datetime.timedelta(minutes=1)
    time = time.strftime('%H%M')
#
# # # Get the volume of each minutes
# result = pd.DataFrame(times,columns=['거래시각'])
#
# day = '20160801'
# while True:
#     try:
#         df = pd.read_csv(f"./raw/kospi_minutes/[지수KOSPI계열]일중 시세정보(1분)(주문번호-2499-1)_{day}.csv", encoding='cp949')
#         df = df[df['지수명']=='코스피']
#         df = df[df['거래시각'] <= '1530']
#         df = df[['거래시각', '거래대금']]
#         df.columns = ['거래시각', f'거래대금_{day}']
#         result = result.join(df.set_index('거래시각'), on='거래시각', how='outer')
#
#     except:
#         print(f"File not found: {day}")
#
#     day = datetime.datetime.strptime(day, '%Y%m%d')
#     day += datetime.timedelta(days=1)
#     day = day.strftime('%Y%m%d')
#
#     # if day == '20240501':
#     if day == '20240501':
#         break
#
#
# result.to_csv('volume.csv', index=False)

# result = pd.read_csv('volume.csv')
# result = result.set_index('거래시각', drop=True)
#
# # 행별 결측치 개수
# result['mean'] = result.mean(axis=1)
# result['std'] = result.std(axis=1)
# result['scaled_mean'] = minmax_scale(result['mean'])
#
# result.reset_index(inplace=True)
#
# result['거래시각'] = times
#
# result.to_csv('volume.csv', index=False)

# length = 0
# length_list = []
# day_list = []
# while True:
#     try:
#         df = pd.read_csv(f"./raw/kospi_minutes/[지수KOSPI계열]일중 시세정보(1분)(주문번호-2499-1)_{day}.csv", encoding='cp949')
#         df = df[df['지수명']=='코스피']
#         df_len = len(df)
#         if length != df_len:
#             length_list.append(df_len)
#             day_list.append(day)
#         length = df_len
#     except:
#         print(f"File not found: {day}")
#
#     day = datetime.datetime.strptime(day, '%Y%m%d')
#     day += datetime.timedelta(days=1)
#     day = day.strftime('%Y%m%d')
#     if day == '20240501':
#         break
#
# change_length_df = pd.DataFrame({'day': day_list, 'length': length_list})
# change_length_df.to_csv('change_length.csv', index=False)
# # pd.set_option('display.max_columns', None)
# # print(df.head())

result = pd.read_csv('volume.csv')
result['proportion'] = result['scaled_mean'] / result['scaled_mean'].sum()
print(result)
