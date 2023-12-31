Designed a BERT-based neural network and RoBERTa model by taking user description, user location, tweet text and other features as input and the number of likes as output to predict the popularity of tweets in climate change in Python. Data is preprocessed from the dataset crawled from the most recent 92540 tweets from different Twitter users using Tweepy. The performance of the model is evaluated with top three important tweet features, to get 90% best accuracy score in prediction.

***relevant_tweet.jsonl*** contains the crawled climate change tweet dataset.

***cleaned_tweet.jsonl*** contains the twitter features after cleaning and preprocessing data.

***bert.py*** contains the Bert-base-uncased model.

***roberta.py*** contains the RoBERTa model.

***pmi.py*** contains the calculation of pointwise mutual information for vocabs.

***feature_experiment.py*** contains feature selection and high-frequency words in our corpus.

To run the two models, first use pip to install pandas, transformers, torch, numpy, nltk, jsonlines, sklearn, tqdm. Then run Python3 bert.py or Python3 roberta.py.

To change selected tweet features for model input, edit selected_keys list and initialize_model function input parameters in main.

To tune hyper parameters, change batch_size in main and adam optimizer parameters in class Tweet2Features, function initialize_model.

To perform pmi calculation, run python3 pmi.py.

To conduct feature experiment, run one of the two functions within python3 feature_experiment.py
