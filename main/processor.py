# CryptoBot/preprocessor.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# load vader
nlp = SentimentIntensityAnalyzer()

class Processor:    
    def __init__(self, data):
        self.data = data

    def process_price_data(self):
        processed_data = {}
        for pair, df in self.data.historical_data.items():
            # Calculate daily returns
            df['Returns'] = df['Close'].pct_change()
            # Calculate the 3-day moving average 
            df['3D_MA'] = df['Close'].rolling(window=3).mean()
            # Calculate the 7-day moving average
            df['7D_MA'] = df['Close'].rolling(window=7).mean()
            # Calculate the 30-day moving average
            df['30D_MA'] = df['Close'].rolling(window=30).mean()
            # Calculate momentum
            df.dropna(inplace=True)
            processed_data[pair] = df

        return processed_data


    def process_news_data(self):
        processed_news = {}

        for crypto, news_data in self.data.historical_news.items():
            processed_articles = []
            for article in news_data:
                polarity = self.get_polarity(article['content'])
                processed_articles.append({'title': article['title'], 
                                           'date': article['date'], 
                                           'polarity': polarity,
                                           'url': article['url']})

            processed_news[crypto] = processed_articles

        return processed_news


    def process_tweets_data(self):
        """
        Calculate tweets polarity, and aggregate per day.
        """
        processed_tweets = {}

        for crypto, tweet_data in self.data.historical_tweets.items():
            processed_tweet_data = []
            for tweet in tweet_data:
                polarity = self.get_polarity(tweet['text'])
                processed_tweet_data.append({'text': tweet['text'], 
                                             'date': tweet['date'], 
                                             'polarity': polarity})

            processed_tweets[crypto] = processed_tweet_data

        return processed_tweets


    def mean_or_zero(self, arr):
        return np.mean(arr) if arr else 0


    def get_polarity(self, text):
        """
        Use spaCy & SpacyTextBlob to get the polarity of a given text.
        """
        doc = nlp.polarity_scores(text)
        return doc['compound']


    def handle_missing_values(self, X, y):
        """
        Remove any rows with missing values in the feature matrix (X) and target vector (y).
        """
        missing_rows = np.isnan(X).any(axis=1)
        X_clean = X[~missing_rows]
        y_clean = y[~missing_rows]

        return X_clean, y_clean


    def aggregate_sentiment_data(self, processed_news_data, processed_tweets_data):
        """
        Aggregate sentiment data (news and tweets) on a per-day basis.
        """
        aggregated_sentiments = {}

        for crypto, news_data in processed_news_data.items():
            sentiments_by_date = {}
            for article in news_data:
                date = article['date'].date()
                if date not in sentiments_by_date:
                    sentiments_by_date[date] = []

                sentiments_by_date[date].append((article['polarity']))

            for tweet in processed_tweets_data[crypto]:
                date = tweet['date'].date()
                if date not in sentiments_by_date:
                    sentiments_by_date[date] = []

                sentiments_by_date[date].append((tweet['polarity']))

            aggregated_sentiments[crypto] = sentiments_by_date

        return aggregated_sentiments


    def create_datasets(self, processed_price_data, processed_news_data, processed_tweets_data, test_size=0.15, val_size=0.15):
        """
        Prepare data for neural network model; aggregate, scale, and split.
        """
        scaler = MinMaxScaler()

        X = []
        y = []

        aggregated_sentiments = self.aggregate_sentiment_data(processed_news_data, processed_tweets_data)

        for pair, df in processed_price_data.items():
            sentiments_by_date = aggregated_sentiments[pair]

            for index, row in df.iterrows():
                date_sentiments = sentiments_by_date.get(index.date(), [])
                news_polarity = np.mean([sentiment[0] for sentiment in date_sentiments]) if date_sentiments else 0

                X.append([row['Close'], row['7D_MA'], row['30D_MA'], news_polarity])
                y.append(row['Returns'])

        # Convert lists to NumPy arrays
        X = np.array(X)
        y = np.array(y)

        # Handle missing values
        X, y = self.handle_missing_values(X, y)

        # Scale the feature vectors
        X = scaler.fit_transform(X)

        return X, y
        # Calculate indices for splitting the dataset
        # dataset_size = len(X)
        # test_split_idx = int(dataset_size * (1 - test_size - val_size))
        # val_split_idx = int(dataset_size * (1 - val_size))

        # # Split the dataset into training, testing, and validation sets sequentially
        # X_train, y_train = X[:test_split_idx], y[:test_split_idx]
        # X_val, y_val = X[test_split_idx:val_split_idx], y[test_split_idx:val_split_idx]
        # X_test, y_test = X[val_split_idx:], y[val_split_idx:]

        # return X_train, X_test, X_val, y_train, y_test, y_val