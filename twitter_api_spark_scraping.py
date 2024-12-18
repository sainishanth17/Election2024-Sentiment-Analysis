import tweepy
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import datetime

# Set up Twitter API credentials
consumer_key = '<redacted_for_privacy>'
consumer_secret = '<redacted_for_privacy>'
access_token = '<redacted_for_privacy>'
access_token_secret = '<redacted_for_privacy>'

# Authenticate to Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Fetch tweets from Twitter API
def fetch_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search, q=query, lang="en", tweet_mode='extended').items(count)
    tweet_data = []
    for tweet in tweets:
        tweet_data.append({
            'created_at': tweet.created_at,
            'text': tweet.full_text,
            'user_join_date': tweet.user.created_at,
            'likes': tweet.favorite_count,
            'retweet_count': tweet.retweet_count,
            'country': tweet.place.country if tweet.place else None
        })
    return pd.DataFrame(tweet_data)

# Fetch tweets for Kamala and Trump
tweets_kamala = fetch_tweets("Kamala", count=100)
tweets_trump = fetch_tweets("Trump", count=100)

# Normalize function
def normalise(x, y):
    x = np.array(pd.to_numeric(x, errors='coerce'))
    y = np.array(y)
    return np.where(x == 0, 0, x / y)

# Sentiment analysis
sid = SIA()
def sentiment(data):
    temp = []
    for row in data:
        tmp = sid.polarity_scores(row)
        temp.append(tmp)
    return temp

# Process date columns
tweets_kamala['user_join_date'] = pd.to_datetime(tweets_kamala['user_join_date'], errors='coerce')
tweets_trump['user_join_date'] = pd.to_datetime(tweets_trump['user_join_date'], errors='coerce')

tweets_kamala['collected_at'] = pd.to_datetime(datetime.datetime.now())
tweets_trump['collected_at'] = pd.to_datetime(datetime.datetime.now())

# Normalize likes and retweets
k_tdiff = (tweets_kamala['collected_at'] - tweets_kamala['created_at']).dt.total_seconds() / 3600
t_tdiff = (tweets_trump['collected_at'] - tweets_trump['created_at']).dt.total_seconds() / 3600

tweets_kamala['likes_norm'] = normalise(tweets_kamala['likes'], k_tdiff)
tweets_kamala['retweet_norm'] = normalise(tweets_kamala['retweet_count'], k_tdiff)

tweets_trump['likes_norm'] = normalise(tweets_trump['likes'], t_tdiff)
tweets_trump['retweet_norm'] = normalise(tweets_trump['retweet_count'], t_tdiff)

# Sentiment scores
tweets_kamala['sentiment'] = sentiment(tweets_kamala['text'])
tweets_trump['sentiment'] = sentiment(tweets_trump['text'])

# Display missing user_join_date counts
print(tweets_kamala['user_join_date'].isnull().sum(), 'missing user_join_date in Kamala data')
print(tweets_trump['user_join_date'].isnull().sum(), 'missing user_join_date in Trump data')

# Print processed data (optional)
print(tweets_kamala.head())
print(tweets_trump.head())
