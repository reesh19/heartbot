# CryptoBot/eda.py
import multiprocessing

import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss


class EDA:
    def __init__(self, X, y, column_names):
        self.X = X
        self.y = y
        self.column_names = column_names
        self.df = pd.DataFrame(X, columns=column_names)
        self.df['Returns'] = y


    def summary_statistics(self):
        return self.df.describe()


    def detect_outliers(self, exclude_cols=None, multiplier=1.5):
        if exclude_cols is None:
            cols = self.column_names
        else:
            cols = [col for col in self.column_names if col not in exclude_cols]

        outlier_data = []

        for col in cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]

            outlier_data.append({
                'column': col,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': outliers.shape[0]
            })

        outliers_df = pd.DataFrame(outlier_data)
        return outliers_df


    def correlation_analysis(self):
        corr_matrix, _ = spearmanr(self.df)
        corr_df = pd.DataFrame(corr_matrix, columns=self.column_names + ['Returns'], index=self.column_names + ['Returns'])
        return corr_df


    def get_vif(self, exclude_cols=None):
        if exclude_cols is None:
            cols = self.column_names
        else:
            cols = [col for col in self.column_names if col not in exclude_cols]

        vif = pd.DataFrame()
        vif["variable"] = cols
        vif["VIF"] = [variance_inflation_factor(self.df[cols].values, i) for i in range(len(cols))]
        return vif


    def _process_column(self, column):
        # check for stationarity using ADF test
        adf_result = adfuller(self.df[column], maxlag=20)
        adf_pvalue = adf_result[1]

        # check for stationarity using KPSS test
        kpss_result = kpss(self.df[column], nlags=20)
        kpss_pvalue = kpss_result[1]

        return {'column': column,
                'adf_pvalue': adf_pvalue,
                'kpss_pvalue': kpss_pvalue}


    def get_adf_kpss(self, exclude_cols=None):
        if exclude_cols is None:
            cols = self.column_names
        else:
            cols = [col for col in self.column_names if col not in exclude_cols]

        # Parallelize column processing using joblib
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self._process_column)(column) for column in cols)

        results_df = pd.DataFrame(results)

        return results_df


    def get_pca(self, exclude_cols=None):
        vif = self.get_vif(exclude_cols=exclude_cols)
        pca_candidates = list(vif[vif.VIF > 10].variable)

        pca = PCA(n_components=0.99)
        X_pca = pca.fit_transform(self.df[pca_candidates])

        data_pca = pd.DataFrame(X_pca, columns=[f"pc_{i + 1}" for i in range(pca.n_components_)])
        non_candidates = [col for col in self.df.columns if col not in pca_candidates]
        data_new = pd.concat([self.df[non_candidates], data_pca], axis=1)

        return data_new


class Transformer:
    def __init__(self, eda: EDA):
        self.eda = eda
        self.df = eda.df.copy()

    def apply_pca(self):
        pca_df = self.eda.get_pca()
        self.df = pca_df.copy()

    def normalize(self):
        scaler = MinMaxScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)

    def apply_differencing(self, adf_threshold=0.05):
        adf_kpss_df = self.eda.get_adf_kpss()
        non_stationary_cols = adf_kpss_df[adf_kpss_df["adf_pvalue"] > adf_threshold]["column"].tolist()

        if non_stationary_cols:
            self.df[non_stationary_cols] = self.df[non_stationary_cols].diff().dropna()

    def transform(self):
        self.apply_pca()
        self.normalize()
        self.apply_differencing()
        return self.df


# CryptoBot/eda.py
import itertools
import multiprocessing

import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss
from torch import nn

# ... Previous EDA class code ...

class EDA:
    # ... Previous EDA class methods ...

    def get_optimal_nn_architecture(self, search_space, X, y, test_size=0.2, cv=3, scoring='neg_mean_squared_error'):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        def create_nn_model(input_dim, hidden_dim, output_dim, dropout):
            model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
            return model

        def train_nn_model(model, X_train, y_train, lr, epochs):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)

            for epoch in range(epochs):
                optimizer.zero_grad()
                predictions = model(X_train_tensor)
                loss = loss_fn(predictions, y_train_tensor)
                loss.backward()
                optimizer.step()

            return model

        def evaluate_nn_model(model, X_test, y_test):
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            with torch.no_grad():
                predictions = model(X_test_tensor)
                mse = mean_squared_error(y_test_tensor, predictions)
            return mse

        def grid_search(params, X_train, y_train, X_test, y_test):
            param_keys, param_values = zip(*params.items())
            best_params = None
            best_mse = float('inf')

            for param_set in itertools.product(*param_values):
                param_dict = dict(zip(param_keys, param_set))
                input_dim = X_train.shape[1]
                output_dim = 1
                model = create_nn_model(input_dim, param_dict['hidden_dim'], output_dim, param_dict['dropout'])
                trained_model = train_nn_model(model, X_train, y_train, param_dict['lr'], param_dict['epochs'])
                mse = evaluate_nn_model(trained_model, X_test, y_test)

                if mse < best_mse:
                    best_mse = mse
                    best_params = param_dict

            return best_params, best_mse

        optimal_params, optimal_mse = grid_search(search_space, X_train, y_train, X_test, y_test)

        return optimal_params, optimal_mse
