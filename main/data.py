# CryptoBot/data.py
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import tweepy
from binance import Client
from binance.exceptions import BinanceAPIException
from bs4 import BeautifulSoup
from dateutil.parser import parse

# Binance API keys and authentication
binance_api_key = os.environ.get('BINANCE_API_KEY')
binance_api_secret = os.environ.get('BINANCE_API_SECRET')
binance = Client(binance_api_key, binance_api_secret)

# Twitter API keys, access tokens, and authentication
consumer_key = os.environ.get('TWITTER_API_KEY')
consumer_secret = os.environ.get('TWITTER_API_SECRET')
access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitter = tweepy.API(auth)

class Data:
    def __init__(self):
        self.cryptos = ['AAVE', 'ALGO', 'BAT', 'BCH', 'BTC', 'DAI', 'ETH', 'GRT', 'LINK', 'LTC', 'MATIC', 'MKR', 'NEAR', 'PAXG', 'SHIB', 'SOL', 'TRX', 'UNI', 'USDT', 'WBTC']
        self.btc_pairs = ['BCH', 'ETH', 'LINK', 'LTC', 'MATIC', 'SOL', 'UNI']
        self.usdt_pairs = ['AAVE', 'ALGO', 'BCH', 'BTC', 'DAI', 'ETH', 'LINK', 'LTC', 'NEAR', 'PAXG', 'SOL', 'TRX', 'UNI']
        self.usd_pairs = ['AAVE', 'ALGO', 'BAT', 'BCH', 'BTC', 'DAI', 'ETH', 'GRT', 'LINK', 'LTC', 'MATIC', 'MKR', 'NEAR', 'PAXG', 'SHIB', 'SOL', 'TRX', 'UNI', 'USDT', 'WBTC']
        self.pairs = self._get_pairs()
        self.historical_data = dict()
        self.historical_news = dict()
        self.historical_tweets = dict()


    def _get_pairs(self):
        return [f'{coin}BTC' for coin in self.btc_pairs] + [f'{coin}USDT' for coin in self.usdt_pairs] + [f'{coin}USD' for coin in self.usd_pairs]


    def _get_klines(self, symbol, interval, start_time, end_time):
        try:
            klines = binance.get_historical_klines(symbol, interval, start_time, end_time)
            return klines
        except BinanceAPIException as e:
            print(f"Error fetching historical klines for {symbol}: {e}")
            return None


    def collect_historical_data(self):
        interval = Client.KLINE_INTERVAL_1HOUR
        start_time = "3 years ago UTC"
        end_time = "now UTC"

        for pair in self.pairs:
            try:
                df = pd.DataFrame(self._get_klines(pair, interval, start_time, end_time))
                df = df.iloc[:, 0:6]
                df.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
                df.set_index('Open time', inplace=True)
                df.index = pd.to_datetime(df.index, unit='ms')
                df = df.astype(float)

            except BinanceAPIException as e:
                print(f"Error fetching historical klines for {pair}: {e}")
                continue

            if df:
                self.historical_data[pair] = df

        return


    def get_crypto_news_urls(self, crypto, start_date, end_date):
        # TODO: Fix this function
        # base_url = "https://cryptonews.com/"
        # https://cryptonews.com/search/?q=Aave
        # search_url = f"{base_url}search/?q={crypto}"
        # page = 1

        # news_urls = []
        # while True:
        #     current_url = f"{search_url}&p={page}"
        #     response = requests.get(current_url)
        #     soup = BeautifulSoup(response.text, "html.parser")

        #     articles = soup.find_all("div", {"class": "search-item__content"})
        #     if not articles:
        #         break

        #     for article in articles:
        #         date_str = article.find("div", {"class": "search-item__date"}).text.strip()
        #         article_date = datetime.strptime(date_str, "%Y-%m-%d")

        #         if start_date <= article_date <= end_date:
        #             news_urls.append(base_url + article.find("a")["href"])
        #         elif article_date < start_date:
        #             return news_urls

        #     page += 1

        # return news_urls
        pass


    def collect_historical_news(self):
        # TODO: Fix this function
        # start_date = datetime.utcnow() - timedelta(days=365 * 3)  # 3 years ago
        # end_date = datetime.utcnow()

        # historical_news = {}

        # for crypto in self.cryptos:
        #     news_urls = self.get_crypto_news_urls(crypto, start_date, end_date)
        #     news_data = []

        #     for url in news_urls:
        #         response = requests.get(url)
        #         soup = BeautifulSoup(response.text, "html.parser")

        #         try:
        #             title = soup.find("h1", {"class": "article__title"}).text.strip()
        #             date_str = soup.find("time", {"class": "article__date"}).text.strip()
        #             date = parse(date_str)
        #             content = soup.find("div", {"class": "article__content"}).text.strip()

        #             news_data.append({"title": title, "date": date, "content": content, "url": url})
        #         except Exception as e:
        #             print(f"Error parsing news article from {url}: {e}")

        #     historical_news[crypto] = news_data

        # return historical_news
        pass


    def collect_historical_tweets(self):
        start_date = datetime.utcnow() - timedelta(days=365 * 3)  # 3 years ago
        end_date = datetime.utcnow()

        for crypto in self.cryptos:
            tweets = tweepy.Cursor(twitter.search_tweets, q=f"{crypto} -filter:retweets", lang="en", tweet_mode='extended', since=start_date.strftime('%Y-%m-%d'), until=end_date.strftime('%Y-%m-%d')).items()
            self.historical_tweets[crypto] = [{"text": tweet.full_text, "date": tweet.created_at} for tweet in tweets]

        return

# Usage
# from CryptoBot import Data

# historical_data, historical_news = self.collect()
# historical_tweets = self.collect_historical_tweets()
