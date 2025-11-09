import numpy as np
import io
import pandas as pd
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from Modules.data_pipeline import DataPipeline
from Modules.column_classifying import ColumnClassifier
from Modules.metric_normalizing import MetricNormalizer

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', None)

mn = MetricNormalizer()

path = "/Users/boychinov/PycharmProjects/Eurotech/Datasets/X_data.csv"

fixed_lines = []
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        s = line.rstrip("\r\n")
        if i == 0:  # baslik
            fixed_lines.append(s + "\n")
            continue

        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]           # dis tirnaklari haric tut
            s = s.replace('""', '"')  # ic tirnaklari düzelt
        fixed_lines.append(s + "\n")

fixed_csv = "".join(fixed_lines)
df = pd.read_csv(io.StringIO(fixed_csv))

df.info()
df.isnull().sum()


### Cols to date
cols_to_date = ['first_purchase_date', 'last_purchase_date', 'last_online_date', 'last_offline_date']

for col in cols_to_date:
    df[col] = pd.to_datetime(df[col])


today = df['last_purchase_date'].max()


df['tenure_days'] = (today - df['first_purchase_date']).dt.days.astype('int64')
#mn.plot_distribution(df["tenure_days"], None, "tenure_days_dagilim")


#df['active_days'] = (df['last_purchase_date'] - df['first_purchase_date']).dt.days.astype('int64')
df['recency'] = (today - df['last_purchase_date']).dt.days.astype('int64')
#mn.plot_distribution(df["recency"], None, "recency_dagilim")

df['spend_total']  = df['spend_offline_total'] + df['spend_online_total']
#mn.plot_distribution(df["spend_total"], None, "spend_total_dagilim")


cols = ["spend_total", "tenure_days", "recency"]
m = df[cols].astype(float).corr(method="pearson")
print(m.round(3))


df_to_scale = df[['recency', 'tenure_days', 'spend_total']]


#orders_total = df['num_orders_online'] + df['num_orders_offline']

#df['spend_per_order'] = (df['spend_total'] / df['orders_total'].replace(0, np.nan)).round(2)

#months = np.maximum(1, df['tenure_days'] / 30.44)

# df['spend_per_month']  = (spend_total  / months).round(2)
# mn.plot_distribution(df["spend_per_month"], None, "spend_per_month_dagilim")
# df['orders_per_month'] = (orders_total / months).round(2)
# mn.plot_distribution(df["orders_per_month"], None, "orders_per_month_dagilim")

pipeline = DataPipeline(df_to_scale)

# Drop Unnecessary Cols
# column_groups = pipeline.classifier.get_column_groups_for_encoding()
# drop_cols = column_groups['drop'] + column_groups['high_cardinality'] + column_groups['datetime'] + column_groups['one_hot']
# pipeline.apply_preprocessing(lambda pp: pp.drop_cols(drop_cols))

pipeline.get_summary()

# Scaling
column_groups = pipeline.classifier.get_column_groups_for_encoding()
scale_cols = column_groups['numerical']
scaler = MinMaxScaler()
pipeline.df[scale_cols] = scaler.fit_transform(pipeline.df[scale_cols])

pipeline.get_summary()

df_scaled = pipeline.df

# df.head(60)
#
# kmeans = KMeans()
# elbow = KElbowVisualizer(kmeans, k=(2, 20))
# elbow.fit(df_scaled)
#
# k = elbow.elbow_value_
# print("k =", k)
#
# plt.figure()
# plt.plot(elbow.k_values_, elbow.k_scores_, marker="o")
# plt.xlabel("k"); plt.ylabel("distortion"); plt.tight_layout()
# plt.savefig("elbow.png")

kmeans = KMeans(n_clusters=3).fit(df_scaled)

# kmeans.n_clusters
# kmeans.cluster_centers_
# kmeans.labels_
# df[0:5]

clusters_kmeans = kmeans.labels_

df['cluster_kmeans'] = clusters_kmeans + 1

# df.tail(60)
df_show = df[['tenure_days', 'recency', 'spend_total', 'cluster_kmeans']]
df_show.groupby('cluster_kmeans').agg(['count', 'mean'])

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import pandas as pd

cols = ['recency','tenure_days','spend_total']
X = df_scaled[cols].to_numpy(copy=False)

km = KMeans(n_clusters=3, n_init=50, random_state=42).fit(X)

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
sil = silhouette_score(X, km.labels_)
db  = davies_bouldin_score(X, km.labels_)
ch  = calinski_harabasz_score(X, km.labels_)
print({'silhouette': round(sil,3), 'davies_bouldin': round(db,3), 'calinski_harabasz': round(ch,1)})









import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

cols = ['recency','tenure_days','spend_total']
X = df[cols].to_numpy()
y = df['cluster_kmeans'].to_numpy()

# 1) ANOVA eta^2 (küme ayırma gücü)
def eta2(x, y):
    m = x.mean()
    ss_tot = ((x - m)**2).sum()
    ss_between = sum((x[y==c].size) * (x[y==c].mean()-m)**2 for c in np.unique(y))
    return 0.0 if ss_tot == 0 else ss_between/ss_tot

eta = {c: eta2(df[c].values, y) for c in cols}

# 2) Karar ağacı önemleri (küme etiketini tahmin ettir)
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(df[cols], y)
tree_imp = dict(zip(cols, clf.feature_importances_))

# 3) Permütasyon – silüet düşüşü (en pratik önem ölçümü)
Xs = MinMaxScaler().fit_transform(df[cols])
km0 = KMeans(n_clusters=df['cluster_kmeans'].nunique(), random_state=42).fit(Xs)
base_sil = silhouette_score(Xs, km0.labels_)

perm_drop = {}
for i, c in enumerate(cols):
    Xp = Xs.copy()
    np.random.shuffle(Xp[:, i])                 # sadece bu özelliği boz
    km = KMeans(n_clusters=km0.n_clusters, random_state=42).fit(Xp)
    perm_drop[c] = base_sil - silhouette_score(Xp, km.labels_)

out = pd.DataFrame({
    'eta2': pd.Series(eta),
    'tree_importance': pd.Series(tree_imp),
    'perm_silhouette_drop': pd.Series(perm_drop)
}).sort_values('perm_silhouette_drop', ascending=False)

print(out.round(3))


############ HC #############

hc_average = linkage(df_scaled, "average")


plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.savefig("hc_plot.png")



plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.savefig("hc2_plot.png")




plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.savefig("hc3_plot.png")


### Model ###
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=4, linkage="average")

hi_clusters = cluster.fit_predict(df_scaled)

df["cluster_hi"] = hi_clusters

df["cluster_hi"] = df["cluster_hi"] + 1

df["cluster_hi"].value_counts()