import pandas as pd
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

control_df = pd.read_excel('/Users/boychinov/PycharmProjects/Eurotech/Datasets/ab_testing_data.xlsx' , sheet_name='Control Group')
test_df = pd.read_excel('/Users/boychinov/PycharmProjects/Eurotech/Datasets/ab_testing_data.xlsx' , sheet_name='Test Group')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

control_df.head()
test_df.head()

control_df.shape
test_df.shape
control_df.dtypes
test_df.dtypes

control_df.agg(['mean', 'median', 'std']).T
test_df.agg(['mean', 'median', 'std']).T

df = pd.concat([control_df, test_df], axis=0).reset_index()

# df.loc[[i for i in range(40,80)], ['index', 'Impression', 'Click', 'Purchase', 'Earning']]

df.head()

# Avg Control Group
df.iloc[0:40, 3:4].mean() # Avg Control = 550.89

# Avg Test Group
df.loc[[i for i in range(40,80)], 'Purchase'].mean() # Avg Test = 582.10

# Ortalmalar arasinda belirgin bir fark var. Hipotezleri kururyoruz.

# H0 - Kontrol grubu ile test grubu satis rakamlari arasinda anlamli bir fark yoktur.
# H1 - Kontrol grubu ile test grubu satis rakamlari arasinda anlamli bir fark vardir.

# Control Group Normallik Testi (> 0.05)
test_stat, pvalue = shapiro(df.iloc[0:40, 3:4]) # pvalue = 0.589 -> NORMAL DAGILIM

# Test Group Normallik Testi (> 0.05)
test_stat, pvalue = shapiro(df.loc[[i for i in range(40,80)], 'Purchase']) # pvalue = 0.154  -> NORMAL DAGILIM

# Varyans Homojenligi Testi (> 0.05)

test_stat, pvalue = levene(df.loc[[i for i in range(0,40)], 'Purchase'], df.loc[[i for i in range(40,80)], 'Purchase']) # VARYANS HOMOJEN

# Dagilimlar Normal, Vsryans Homojen -> Prametrik Test (> 0.05)

test_stat, pvalue = ttest_ind(df.loc[df.index < 40, 'Purchase'],     # pvalue = 0.349 -> H0 DOGRU (RED EDILEMEZ)
                              df.loc[df.index >= 40 , 'Purchase'],
                              equal_var=True)

# Dagilimlar Normal Degil, Vsryans Homojen Degil (Olsaydi) -> Non-Parametrik Test (< 0.05) Mannwhitneyu uygulayacaktik.

# SONUC

# Kontrol ve test grupları arasında anlamlı bir fark yoktur.
# Bu sonuçlara göre "Average Bidding" stratejisi, "Maximum Bidding" stratejisine göre anlamlı bir iyileşme sağlamıyor.
# "Maximum Bidding" stratejisi daha etkili görünüyor.


for i in range(1000000000):
    i=i+1
    print(i)
