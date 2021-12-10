import tensorflow as tf
import numpy as np
import datetime
import time
import tweepy
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import glob
from textblob import TextBlob
import pandas as pd

def get_df():
    """
    Generates a new dataframe.

    :param: None
    :return: the new dataframe
    """
    return pd.DataFrame(
        columns=[
            "tweet_id",
            "name",
            "screen_name",
            "retweet_count",
            "text",
            "mined_at",
            "created_at",
            "favourite_count",
            "hashtags",
            "status_count",
            "followers_count",
            "location",
            "source_device",
        ]
    )

"""Class that helps mine for tweets"""
class TweetMiner(object):

    result_limit = 20
    data = []
    api = False

    twitter_keys = {
        "consumer_key": "X",
        "consumer_secret": "X",
        "access_token_key": "X",
        "access_token_secret": "X",
    }

    def __init__(self, keys_dict=twitter_keys, api=api):
        """
        Initializes a new TweetMiner object.

        :param: twitter keys foro authentication and the api being used
        :return: None
        """
        self.twitter_keys = keys_dict

        auth = tweepy.OAuthHandler(
            keys_dict["consumer_key"], keys_dict["consumer_secret"]
        )
        auth.set_access_token(
            keys_dict["access_token_key"], keys_dict["access_token_secret"]
        )
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        self.twitter_keys = keys_dict

    def mine_crypto_currency_tweets(self, query="BTC"):
        """
        Scrapes twitter for all recent mentions of the given query and stores them in a csv file.
        :param query: The cryptocurrency that is being searched for
        :return: None
        """
        last_tweet_id = False
        page_num = 1

        data = get_df()
        cypto_query = f"#{query}"
        print(" ===== ", query, cypto_query)
        for page in tweepy.Cursor(
            self.api.search_tweets,
            q=cypto_query,
            lang="en",
            tweet_mode="extended",
            count=10,  # max_id=1295144957439690000
        ).pages(100):
            print(" ...... new page", page_num)
            page_num += 1

            for item in page:
                mined = {
                    "tweet_id": item.id,
                    "name": item.user.name,
                    "screen_name": item.user.screen_name,
                    "retweet_count": item.retweet_count,
                    "text": item.full_text,
                    "mined_at": datetime.datetime.now(),
                    "created_at": item.created_at,
                    "favourite_count": item.favorite_count,
                    "hashtags": item.entities["hashtags"],
                    "status_count": item.user.statuses_count,
                    "followers_count": item.user.followers_count,
                    "location": item.place,
                    "source_device": item.source,
                }

                try:
                    mined["retweet_text"] = item.retweeted_status.full_text
                except:
                    mined["retweet_text"] = "None"

                last_tweet_id = item.id
                data = data.append(mined, ignore_index=True)

            if page_num % 100 == 0:
                date_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                print("....... outputting to csv ", query)
                data.to_csv(f"{query}.csv", index=False)
                print("  ..... resetting df")
                data = get_df()


def generate_csvs(handle_list):
    """
        For all cryptocurrencies in a list, goes through twitter and adds all the tweets to different CSV files
        :param handle_list: The list of cryptocurrencies that we want to search for occurences of
        :return: None
    """
    miner = TweetMiner()
    for name in handle_list:
        miner.mine_crypto_currency_tweets(name)

def process_csv(curr):
    """
        Converts CSV files to panda dataframes.
        :param curr: The CSV file that is being converted
        :return: newly generated dataframe
    """
    pd_df = pd.concat(
        [pd.read_csv("./data/twitter-data/" + curr + ".csv")],
        ignore_index=True,
    )

    print(f"Number of records loaded for {curr}", pd_df.size)
    # pd_df[["coin_symbol", "tweet_id", "created_at", "date", "hour"]].to_csv(
    #     f"./data/{curr}.csv", index=False
    # )

    return pd_df
def preprocess_text(text):
    """
        Preprocesses the text in a dataframe to prepare it for sentiment analysis.
        :param text: A panda column that is being modified to prepare it for sentiment analysis
        :return: Preprocessed text in an array
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    words = [lemmatizer.lemmatize(w) for w in text if w not in stop_words]
    stem_text = " ".join([stemmer.stem(i) for i in words])
    return stem_text

def get_sentiment(tweet):
    """
        Returns the sentiment of the given tweet.
        :param tweet: Tweet that we want the sentiment of
        :return: Sentiment of the given tweet as a float
    """
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity  # work out sentiment


def get_average_sentiment(crypto):
    """
        Gets the average sentiment of a group of tweets associated with a given crypto.
        :param crypto: The cryptocurrency we want the average sentiment of
        :return: average sentiment of the given crypto
    """
    dataframe = process_csv(crypto)
    dataframe.drop_duplicates(
        subset=["tweet_id"], inplace=True, keep="last", ignore_index=True
    )
    
    tokenizer = RegexpTokenizer(r"\w+")
    dataframe["processed text"] = dataframe["text"].apply(
        lambda x: preprocess_text(tokenizer.tokenize(str(x).lower()))
    )
    dataframe["polarity"] = dataframe["processed text"].apply(lambda x: get_sentiment(x))
    average_polarity = dataframe["polarity"].mean()
    print(dataframe.head(10))
    return average_polarity

handle_list = [
    "ADA",
    "AVAX",
    "BNB",
    "BTC",
    "DOGE",
    "ETH",
    "DOT",
    "LUNA",
    "SOL",
    "XRP"
]
"""Code is for getting tweets and saving them to a CSV file"""
# for crypto in handle_list:
#     miner = TweetMiner()
#     miner.mine_crypto_currency_tweets(crypto)

"""Code is for calculating and printing the average sentiment from each of the different cryptos"""
for crypto in handle_list:
    get_average_sentiment(crypto)
    # print(get_average_sentiment(crypto))


