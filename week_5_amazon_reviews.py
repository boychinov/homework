import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
import Modules.data_processor as dp

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('/Users/boychinov/PycharmProjects/Eurotech/Datasets/amazon_reviews.csv')

data_pro = dp.DataProcessor(df)
data_pro.check_df()

df['overall'].value_counts()
avg = df['overall'].mean()
df['day_diff'].max()
df.tail()
df.shape
avg
df.loc[(df['day_diff'] > 0) & (df['day_diff'] <= 365), 'overall'].mean() # Son 1 yil
df[(df['day_diff'] > 365) & (df['day_diff'] <= 730)]['overall'].mean()   # Bir önceki yil
df[(df['day_diff'] > 730) & (df['day_diff'] <= 1064)]['overall'].mean()  # 2 yil önceki yil

df[df['day_diff'] <= 30].mean()
df.loc[(df['day_diff'] > 30) & (df['day_diff'] <= 90), 'overall'].mean()
df[(df['day_diff'] > 90) & (df['day_diff'] <=180)]['overall'].mean()
df.loc[df['day_diff'] > 180, 'overall'].mean()

def time_based_weighted_average():

    return df[df['day_diff'] <= 30]['overall'].mean() * 28/100 + \
           df.loc[(df['day_diff'] > 30) & (df['day_diff'] <= 90), 'overall'].mean() * 26/100 + \
           df[(df['day_diff'] > 90) & (df['day_diff'] <= 180)]['overall'].mean() * 24/100 + \
           df.loc[df['day_diff'] > 180, 'overall'].mean() * 22/100



def weighted_average_time_based(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

dbw_avg = time_based_weighted_average()
dbw_avg
df.head(130)