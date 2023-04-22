import datetime
import json
import os
import math

# import spacy
import tweepy
from binance import ThreadedWebsocketManager
from dotenv import load_dotenv
from newsapi import NewsApiClient
# from spacytextblob.spacytextblob import SpacyTextBlob
from tweepy import OAuthHandler, Stream, StreamingClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
load_dotenv()

# Binance API keys and authentication
binance_api_key = os.environ.get('BINANCE_API_KEY')
binance_api_secret = os.environ.get('BINANCE_API_SECRET')

twm = ThreadedWebsocketManager(api_key=binance_api_key, api_secret=binance_api_secret)

# Twitter API keys, access tokens, and authentication
tw_consumer_key = os.environ.get('TWITTER_API_KEY')
tw_consumer_secret = os.environ.get('TWITTER_API_SECRET')
tw_access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
tw_access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')

tw_auth = OAuthHandler(tw_consumer_key, tw_consumer_secret)
tw_auth.set_access_token(tw_access_token, tw_access_token_secret)

# News API key and client
news_api_key = os.environ.get('NEWS_API_KEY')
news = NewsApiClient(api_key=news_api_key)

# load vader sentiment analyzer
nlp = SentimentIntensityAnalyzer()

class LiveData:
    def __init__(self):
        self.cryptos = {
            "AAVE": '("AAVE" OR "Aave" OR "aave") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "ALGO": '("ALGO" OR "Algorand" OR "algorand") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "BAT": '("BAT" OR "Basic Attention Token" OR "basic-attention-token") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "BCH": '("BCH" OR "Bitcoin Cash" OR "bitcoin-cash") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "BTC": '("BTC" OR "Bitcoin" OR "bitcoin") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "DAI": '("DAI" OR "Dai" OR "multi-collateral-dai") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "ETH": '("ETH" OR "Ethereum" OR "ethereum") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "GRT": '("GRT" OR "The Graph" OR "the-graph") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "LINK": '("LINK" OR "Chainlink" OR "chainlink") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "LTC": '("LTC" OR "Litecoin" OR "litecoin") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "MATIC": '("MATIC" OR "Polygon" OR "polygon") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "MKR": '("MKR" OR "Maker" OR "maker") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "NEAR": '("NEAR" OR "NEAR Protocol" OR "near-protocol") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "PAXG": '("PAXG" OR "PAX Gold" OR "pax-gold") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "SHIB": '("SHIB" OR "Shiba Inu" OR "shiba-inu") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "SOL": '("SOL" OR "Solana" OR "solana") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "TRX": '("TRX" OR "TRON" OR "tron") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "UNI": '("UNI" OR "Uniswap" OR "uniswap") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "USDT": '("USDT" OR "Tether" OR "tether") AND ("crypto" OR "CRYPTO" OR "Crypto")',
            "WBTC": '("WBTC" OR "Wrapped Bitcoin" OR "wrapped-bitcoin") AND ("crypto" OR "CRYPTO" OR "Crypto")',
        }
        
        self.btc_pairs = ['BCH', 'ETH', 'LINK', 'LTC', 'MATIC', 'SOL', 'UNI']
        self.usdt_pairs = ['AAVE', 'ALGO', 'BCH', 'BTC', 'DAI', 'ETH', 'LINK', 'LTC', 'NEAR', 'PAXG', 'SOL', 'TRX', 'UNI']
        self.usd_pairs = ['AAVE', 'ALGO', 'BAT', 'BCH', 'BTC', 'DAI', 'ETH', 'GRT', 'LINK', 'LTC', 'MATIC', 'MKR', 'NEAR', 'PAXG', 'SHIB', 'SOL', 'TRX', 'UNI', 'USDT', 'WBTC']
        
        self.pairs = self._get_pairs()
        
        self.price_data = dict()
        self.tweets_data = dict()
        self.news_data = dict()


    def _get_pairs(self):
        return [f'{coin}BTC' for coin in self.btc_pairs] + [f'{coin}USDT' for coin in self.usdt_pairs] + [f'{coin}USD' for coin in self.usd_pairs]


    def _get_sentiment(self, text):
        """
        Use VaderSentiment to get compound polarity of a text input.
        """
        doc = nlp.polarity_scores(text)
        return doc['compound']


    def _on_price_update(self, msg):
        if msg['e'] == 'trade':
            symbol = msg['s']
            crypto = symbol[:-4]
            if crypto in self.cryptos:
                price = float(msg['p'])
                timestamp = msg['T']
                self.price_data[crypto] = {'price': price, 'timestamp': timestamp}


    def start_price_stream(self):
        """
        Start a price stream for each crypto in self.cryptos.keys().
        """
        for pair in self.pairs:
            twm.start_trade_socket(callback=self._on_price_update, symbol=pair)
        
        twm.join()


    class TwitterListener(StreamingClient):
        def __init__(self):
            super().__init__(bearer_token=tw_access_token, wait_on_rate_limit=True)


        def on_data(self, data):
            try:
                tweet_data = json.loads(data)
                self.on_tweet(tweet_data)
            except Exception as e:
                print(f'Error: {e}')
            return True


        def on_error(self, status):
            print(status)


        def on_tweet(self, tweet_data):
            # TODO 
            pass


    def start_twitter_stream(self):
        """
        Start a tweets stream for each crypto in self.cryptos.keys().
        Use relevant # or $ tags to collect tweets for each crypto.
        Treat retweets as an additional tweet, but only count 1 tweet or retweet per user account.
        Use _get_sentiment() to get the sentiment of each tweet as it is received and save it in self.tweets_data.
        """
        twitter_listener = self.TwitterListener()
        twitter_stream = Stream(consumer_key=tw_consumer_key,
                                consumer_secret=tw_consumer_secret,
                                access_token=tw_access_token,
                                access_token_secret=tw_access_token_secret,
                                listener=twitter_listener)
        
        # TODO: get tweets for all cryptos simultaneously 
        # This might be wrong, but it's just an idea:
        # twitter_stream.filter(track=self.cryptos[crypto].split(' OR '), languages=['en'])
        # tweets should be aggregated on a 1min increments to match the frequency of the price data stream, and sentiment analaysis can be run on the aggregated tweets using _get_sentiment() from processor.py.


    def get_live_sentiment_news(self):
        """
        Collect news for each crypto in self.cryptos.keys() using self.cryptos.values() as queries.
        Use _get_sentiment() to get the sentiment of each article's title + description, and save it in self.news_data.
        """
        dtformat = "%Y-%m-%d"
        from_param = datetime.datetime.now() - datetime.timedelta(1)
        from_param = from_param.strftime(dtformat)
        for crypto, search_string in self.cryptos.items():
            response = news.get_everything(q=search_string, from_param=from_param, language='en')
            data = [self._get_sentiment(f"{i['title']} {i['description']}") for i in response['articles']]
            self.news_data[crypto] = sum(data) / len(data)

        return

    def start_live_data_inference(self):
        self.start_price_stream()
        self.start_twitter_stream()
        self.get_live_sentiment_news()



# if __name__ == '__main__':
#     # print all keys
#     print('Keys:')
#     print(f'  Twitter consumer key: {tw_consumer_key}')
#     print(f'  Twitter consumer secret: {tw_consumer_secret}')
#     print(f'  Twitter access token: {tw_access_token}')
#     print(f'  Twitter access token secret: {tw_access_token_secret}')
#     print(f'  News API key: {news_api_key}')
#     print() 
#     print(f'  Binance API key: {binance_api_key}')
#     print(f'  Binance API secret: {binance_api_secret}')
#     print()
#     print(f'  NewsAPI key: {news_api_key}')



# import json
# import tweepy

# class TweetListener(tweepy.StreamingClient):
#     """Handles incoming Tweet stream."""

#     def __init__(self, bearer_token, database, limit=10000):
#         """Create instance variables for tracking number of tweets."""
#         self.db = database
#         self.tweet_count = 0
#         self.TWEET_LIMIT = limit  # 10,000 by default
#         super().__init__(bearer_token, wait_on_rate_limit=True)  

#     def on_connect(self):
#         print('Twitter connection successful\n')

#     def on_data(self, data):
#         """Called when Twitter pushes a new tweet to you."""
#         self.tweet_count += 1  # track number of tweets processed
#         data = json.loads(data)  # convert string to JSON
        
#         # TODO
        
#         if self.tweet_count == self.TWEET_LIMIT:
#             self.disconnect()
    
#     def on_exception(self, status):
#         print(f'Error: {status}')
#         return True