import jsonlines
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math

text = []
like_text = []

text_vocab = 0
char1_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,1))

with jsonlines.open('cleaned_tweet.jsonl') as reader:
  for line in reader:
    if line["num_of_likes"] > 100 and line["num_of_likes"] < 4000:
      text.append(line["text"])
      text_vocab += len(line["text"].split())
    if line["num_of_likes"] >= 356 and line["num_of_likes"] < 4000:
      like_text.append(line["text"])

train_1 = char1_vectorizer.fit_transform(text).toarray()
train_1_names = char1_vectorizer.get_feature_names()

train_2 = char1_vectorizer.fit_transform(like_text).toarray()
train_2_names = char1_vectorizer.get_feature_names()

train_1 = np.array(train_1)
train_2 = np.array(train_2)
# train_1[train_1 > 0] = 1
# train_2[train_2 > 0] = 1

freq_1 = np.sum(train_1, axis=0)
freq_2 = np.sum(train_2, axis=0)

p_b = 0.5
results = []
for index, item in enumerate(train_1_names):
  p_a = freq_1[index]/text_vocab
  p_ab = 0
  if item in train_2_names:
    p_ab = freq_2[train_2_names.index(item)]/text_vocab
  PMI = math.log(p_ab/(p_a*p_b)+1)
  results.append(PMI)

results = np.array(results)
sorted_index_array = np.argsort(results)
n = 10
rslt = sorted_index_array[-n : ]
for item in rslt:
  print(train_1_names[item])
