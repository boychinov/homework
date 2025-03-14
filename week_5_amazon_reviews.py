import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('/Users/boychinov/PycharmProjects/Eurotech/Datasets/amazon_reviews.csv')

# GÖREV 1

df['overall'].value_counts()
avg = df['overall'].mean()
df['day_diff'].max()
df.tail()
df.shape
avg
df.loc[(df['day_diff'] > 0) & (df['day_diff'] <= 365), 'overall'].mean() # Son 1 yil
df[(df['day_diff'] > 365) & (df['day_diff'] <= 730)]['overall'].mean()   # Bir önceki yil
df[(df['day_diff'] > 730) & (df['day_diff'] <= 1064)]['overall'].mean()  # 2 yil önceki yil

df[df['day_diff'] <= 30]['overall'].mean()
df.loc[(df['day_diff'] > 30) & (df['day_diff'] <= 90), 'overall'].mean()
df[(df['day_diff'] > 90) & (df['day_diff'] <=180)]['overall'].mean()
df.loc[df['day_diff'] > 180, 'overall'].mean()

def time_based_weighted_average():

    return df[df['day_diff'] <= 30]['overall'].mean() * 28/100 + \
           df.loc[(df['day_diff'] > 30) & (df['day_diff'] <= 90), 'overall'].mean() * 26/100 + \
           df[(df['day_diff'] > 90) & (df['day_diff'] <= 180)]['overall'].mean() * 24/100 + \
           df.loc[df['day_diff'] > 180, 'overall'].mean() * 22/100


dbw_avg = time_based_weighted_average()
dbw_avg
avg

# GÖREV 2

df['helpful_no'] = df.apply(lambda x: x['total_vote'] - x['helpful_yes'], axis=1)
# df['helpful_no'] = df['total_vote'] - df['helpful_yes']
df.head(150)

def up_down_diff(up, down):
    return up - down

df['pos_neg_diff'] = df.apply(lambda x: up_down_diff(x['helpful_yes'], x['helpful_no']), axis=1)
df.head(50)

def up_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df['average_rating_score'] = df.apply(lambda x: up_rating(x['helpful_yes'], x['helpful_no']), axis=1)
df.head(50)

def wilson_lower_bound(up, down, confidence=0.95):

    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n

    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1)

df.sort_values('wilson_lower_bound', ascending=False).head(20)