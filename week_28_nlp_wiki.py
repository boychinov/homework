from Modules.nlp import Nlp
import pandas as pd
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv('/Homework/Wiki/wiki_data.csv', encoding='utf-8', encoding_errors='ignore')
df = df.drop('Unnamed: 0', axis=1)

nlp = Nlp(df, 'text')

nlp = Nlp(df, 'text').clean_basic(stop=True)
nlp.df
tf_after_cleaning = nlp.term_frequency()

nlp.remove_rare_words(max_freq=50)
nlp.df
tf_after_rare = nlp.term_frequency()


nlp.tokenize_words().lemmatize_tokens(pos='v')
#nlp.df
tf_after_tok_lem = nlp.term_frequency()
tf_after_tok_lem.head(40)

nlp.remove_stopwords(extra={'would', 'one', 'also', 'may', 'u', 'e'})

tf_after_tok_lem_stop = nlp.term_frequency()
tf_after_tok_lem_stop.head(50)


nlp.plot_term_frequency(top_n=50, rotate=90)


nlp.plot_wordcloud(
    max_words=50,
    max_font_size=100,
    background_color="black",
    width=1200, height=600,
    colormap="turbo", #  "plasma" / "turbo" / "magma" / "inferno"
    save_path="/Training/wordcloud.png",
    show=True
)


