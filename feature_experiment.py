import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def count_freq():
    df = pd.read_json('cleaned_tweet.jsonl', lines=True)
    df.drop_duplicates(subset ="text", keep = False, inplace = True)
    # loss_fn = nn.CrossEntropyLoss()

    df = df[df['num_of_likes'] < 4000]
    df = df[df['num_of_likes'] > 100]
    # df['num_of_likes'].describe()
    df['num_of_likes'] = (df['num_of_likes'] >= 356).astype(int)

    corpus = []
    for index, row in df.iterrows():
        corpus.append(row[4])


    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(ngram_range=(1,1),tokenizer=token.tokenize, max_features=15)
    text_count = cv.fit_transform(corpus).toarray()
    freq_sum = np.sum(text_count, axis=0)
    featured_word = cv.get_feature_names()

def feature_selection(train_df, k):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    target_col = 'num_of_likes'
    x = train_df[['follower_count', 'friends_count', 'retweet_count', 'has_link', 'num_emoji', 'num_tag', 'num_at', 'has_num', 'has_mark']]
    y = train_df[target_col]
    select = SelectKBest(score_func=chi2, k=)
    z = select.fit_transform(x,y)
    print(z)


