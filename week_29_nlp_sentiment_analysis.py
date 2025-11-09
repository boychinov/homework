import matplotlib
matplotlib.use('MacOSX')
from Modules.nlp import Nlp

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.pipeline import make_pipeline

from nltk.sentiment import SentimentIntensityAnalyzer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', 200)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_excel('/Users/boychinov/PycharmProjects/Eurotech/Homework/Nlp_Sentiment_Analysis/amazon.xlsx')

# Veriseti inceleme
df.head(10)
df.shape
df.isna().sum()

df = df[['Review']].dropna(subset=['Review'])
df['Review'] = df['Review'].astype(str)
df.shape

######### 1-2 Metin Önisleme, Görsellestirme ##########
nlp = Nlp(df, 'Review')
nlp.set_data(df)


# Temizle
nlp.clean_basic(stop=True, bind_negs=True)
df.shape
nlp.df.shape
nlp.df

# Tf
tf_after_cleaning = nlp.term_frequency()
tf_after_cleaning.head(20)
tf_after_cleaning.shape


# Rare Cikar
nlp.remove_rare_words(max_freq=5)
tf_after_rare = nlp.term_frequency()
tf_after_rare.tail(20)
tf_after_rare.shape

# Tokenize & Lemmatize
nlp.tokenize_words().lemmatize_tokens(pos='n')
nlp.df
tf_after_tok_lem = nlp.term_frequency()
tf_after_tok_lem.head(20)
tf_after_tok_lem.shape

# Bar Plot
nlp.plot_term_frequency(top_n=100,  rotate=90, min_tf=500)

# Wordcloud
nlp.plot_wordcloud(
    max_words=1500,
    max_font_size=200,
    background_color="black",
    width=1400, height=700,
    colormap="turbo", #  "plasma" / "turbo" / "magma" / "inferno"
    save_path="/Users/boychinov/PycharmProjects/Eurotech/Training/wordcloud.png",
    show=True
)


############ 3 Duygu Analizi ###############
nlp_sent = Nlp(df, 'Review')
#nlp_sent.set_data(df)

# Basit temizlik & Sadece Ingilizce satirlari al
nlp_sent.normalize_unicode().unescape_html().remove_html_tags().remove_urls().keep_english_rows()

nlp_sent.df.shape

# Tahmin ici hazirlik
sia = SentimentIntensityAnalyzer()
df_sent = nlp_sent.df

# Ilk 10
df_sent_10 = df_sent["Review"][0:10]
df_sent_10 = df_sent_10.to_frame(name='Review')
df_sent_10['compound'] = df_sent_10['Review'].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])
df_sent_10['label'] = df_sent_10['compound'].apply(lambda x: 'pos' if x > 0 else 'neg')
df_sent_10

# Label Col
nlp_sent.add_vader_sentiment()
nlp_sent.df.head(10)

nlp_sent.df['vader_label'].value_counts()
nlp_sent.df['vader_label'].value_counts() / nlp_sent.df.shape[0] * 100


######### 4 ML Hazirlik ###########
df_ml = nlp_sent.df[['Review', 'vader_label']]

X = df_ml['Review']
y = df_ml['vader_label']

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=42,stratify=y)

# Vektörize et
vectorizer = TfidfVectorizer()

vectorizer.fit(X_train)

X_train_vec = vectorizer.transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print("X_train_vec shape:", X_train_vec.shape)
print("X_test_vec  shape:", X_test_vec.shape)


###### 5 Log Reg

# Model
log_reg = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced', random_state=42)
log_reg.fit(X_train_vec, y_train)

# Tahmin
y_pred = log_reg.predict(X_test_vec)

# Rapor
print(classification_report(y_test, y_pred, digits=3))

# Cross Val
pipe_log = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced', random_state=42)
)
cv_scores_log = cross_val_score(pipe_log, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print("LOGREG CV accuracy (foldlar):", np.round(cv_scores_log, 4))
print("LOGREG CV accuracy (ort):    ", cv_scores_log.mean().round(4))


# cv_scores = cross_val_score(log_reg, X_train_vec, y_train, cv=5, scoring='accuracy')
# print("CV accuracy (fold bazında):", np.round(cv_scores, 4))
# print("CV accuracy (ortalama):    ", cv_scores.mean().round(4))

# Örnek al
sample_texts = df_ml['Review'].sample(5, random_state=17)

# Vektörize et
sample_vec = vectorizer.transform(sample_texts)

# Tahmin
sample_pred = log_reg.predict(sample_vec)
sample_pred

# Sayisal veriyi etiketle
label_map = {0: 'neg', 1: 'pos'}
sample_pred_labels = [label_map[int(p)] for p in sample_pred]
sample_pred_labels

# Print
for txt, lab in zip(sample_texts, sample_pred_labels):
    print("—")
    print("Review:", txt[:300].replace("\n", " "))
    print("Tahmin:", lab)


######## 6 RF

# Model
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train_vec, y_train)

# Tahmin
y_pred_rf = rf.predict(X_test_vec)

# Rapor
print(classification_report(y_test, y_pred_rf, digits=3))

# Cross Val
pipe_rf = make_pipeline(
    TfidfVectorizer(),
    RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1)
)
cv_scores_rf = cross_val_score(pipe_rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print("RF CV accuracy (foldlar):", np.round(cv_scores_rf, 4))
print("RF CV accuracy (ort):    ", cv_scores_rf.mean().round(4))


# cv_scores_rf = cross_val_score(rf, X_train_vec, y_train, cv=5, scoring='accuracy', n_jobs=-1)
# print("RF CV accuracy (foldlar):", np.round(cv_scores_rf, 4))
# print("RF CV accuracy (ortalama):", cv_scores_rf.mean().round(4))


# Model Kiyaslama

acc_log = accuracy_score(y_test, y_pred)
acc_rf  = accuracy_score(y_test, y_pred_rf)

cmp = pd.DataFrame({
    'Logistic':     [acc_log,
                     recall_score(y_test, y_pred, pos_label=0),
                     recall_score(y_test, y_pred, pos_label=1)],
    'RandomForest': [acc_rf,
                     recall_score(y_test, y_pred_rf, pos_label=0),
                     recall_score(y_test, y_pred_rf, pos_label=1)]
}, index=['Accuracy', 'Recall_0 (neg)', 'Recall_1 (pos)'])

print(cmp)