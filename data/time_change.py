import pandas as pd
import datetime

# # # Load data
# # day = '20000104'
# day = '20160801'
#
# times = []
# time = '0900'
# while time != '1531':
#     times.append(time)
#     time = datetime.datetime.strptime(time, '%H%M')
#     time += datetime.timedelta(minutes=1)
#     time = time.strftime('%H%M')
#
# # # Get the volume of each minutes
# result = pd.DataFrame(times,columns=['거래시각'])
#
# while True:
#     try:
#         df = pd.read_csv(f"./raw/kospi_minutes/[지수KOSPI계열]일중 시세정보(1분)(주문번호-2499-1)_{day}.csv", encoding='cp949')
#         df = df[df['지수명']=='코스피']
#         df = df[df['거래시각'] <= '1530']
#         df = df[['거래시각', '거래대금']]
#         result = result.join(df.set_index('거래시각'), on='거래시각', how='outer', rsuffix=f'_{day}')
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
# print(result)
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