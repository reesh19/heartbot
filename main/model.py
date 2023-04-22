import itertools
import os
import time

import schedule
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset


class CNNLSTM(nn.Module):
    def __init__(self, input_size, cnn_output_size, lstm_hidden_size, num_layers, output_size):
        super(CNNLSTM, self).__init__()

        self.input_size = input_size
        self.cnn_output_size = cnn_output_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # CNN layers
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()

        # LSTM layers
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size, output_size)


    def forward(self, x):
        # Pass the input through the CNN layers
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))

        # Flatten the output and prepare it for the LSTM layers
        x = self.flatten(x)
        x = x.view(-1, 1, 64)

        # Pass the output through the LSTM layers
        lstm_out, _ = self.lstm(x)

        # Pass the LSTM output through the fully connected layer
        out = self.fc(lstm_out[:, -1, :])

        return out


class CryptoModel:
    def __init__(self, data, eda, transformer):
        self.data = data
        self.eda = eda
        self.transformer = transformer
        self.model = None
        self.criterion = torch.nn.MSELoss()


    def prepare_data(self, test_size=0.15, val_size=0.15):
        transformed_data = self.transformer.transform()
        X = transformed_data.drop(columns=['Returns']).values
        y = transformed_data['Returns'].values

        # Split the data into train, validation, and test sets
        
        dataset_size = len(X)
        test_split_idx = int(dataset_size * (1 - test_size - val_size))
        val_split_idx = int(dataset_size * (1 - val_size))

        # Split the dataset into training, testing, and validation sets sequentially
        train_X, train_y = X[:test_split_idx], y[:test_split_idx]
        val_X, val_y = X[test_split_idx:val_split_idx], y[test_split_idx:val_split_idx]
        test_X, test_y = X[val_split_idx:], y[val_split_idx:]
        

        # Convert the data to PyTorch tensors
        train_X, val_X, test_X = torch.tensor(train_X, dtype=torch.float32), torch.tensor(val_X, dtype=torch.float32), torch.tensor(test_X, dtype=torch.float32)
        train_y, val_y, test_y = torch.tensor(train_y, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)

        # Create PyTorch data loaders
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        test_dataset = TensorDataset(test_X, test_y)

        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


    def train(self, train_loader, val_loader, epochs=100, learning_rate=0.001):
        input_size = self.transformer.df.shape[1] - 1
        cnn_output_size = 64
        lstm_hidden_size = 128
        num_layers = 2
        output_size = 1

        # Initialize the CNN-LSTM model
        self.model = CNNLSTM(input_size, cnn_output_size, lstm_hidden_size, num_layers, output_size)

        # Use Mean Squared Error as the loss function
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            train_loss = 0.0
            self.model.train()
            for inputs, targets in train_loader:
                # Make sure the input data has the correct shape for the CNN-LSTM model
                inputs = inputs.permute(0, 2, 1)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets.view(-1, 1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Evaluate the model on the validation set
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.permute(0, 2, 1)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.view(-1, 1))
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')


    def evaluate(self, test_loader):
        test_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.permute(0, 2, 1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.view(-1, 1))
                test_loss += loss.item()

        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')


    def evaluate_validation(self, val_loader):
        val_loss = 0.0
        self.model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.permute(0, 2, 1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.view(-1, 1))
                val_loss += loss.item()
                
                val_preds.extend(outputs.view(-1).tolist())
                val_targets.extend(targets.view(-1).tolist())

        val_loss /= len(val_loader)
        mse = mean_squared_error(val_targets, val_preds)
        mae = mean_absolute_error(val_targets, val_preds)
        r2 = r2_score(val_targets, val_preds)
        print(f'Validation Loss: {val_loss:.4f}, Mean Squared Error: {mse:.4f}, Mean Absolute Error: {mae:.4f}, R-squared: {r2:.4f}')


    def grid_search(self, train_loader, val_loader, param_grid):
        best_val_loss = float('inf')
        best_params = None

        for params in itertools.product(*param_grid.values()):
            hyperparameters = dict(zip(param_grid.keys(), params))
            print(f'Training with hyperparameters: {hyperparameters}')

            # Update model's hyperparameters
            learning_rate = hyperparameters.get('learning_rate', 0.001)
            cnn_output_size = hyperparameters.get('cnn_output_size', 64)
            lstm_hidden_size = hyperparameters.get('lstm_hidden_size', 128)
            num_layers = hyperparameters.get('num_layers', 2)
            epochs = hyperparameters.get('epochs', 100)

            # Initialize the CNN-LSTM model
            self.model = CNNLSTM(self.input_size, cnn_output_size, lstm_hidden_size, num_layers, self.output_size)

            # Train the model with the current hyperparameters
            self.train(train_loader, val_loader, epochs=epochs, learning_rate=learning_rate, verbose=False)

            # Evaluate the model on the validation set
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.permute(0, 2, 1)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets.view(-1, 1))
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')

            # Update the best parameters if the current configuration has a lower validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = hyperparameters

        print(f'Best hyperparameters found: {best_params}')
        return best_params


    def optimize(self, test_loader):
        # Define the hyperparameters grid
        param_grid = {
            'learning_rate': [0.01, 0.001, 0.0001],
            'cnn_output_size': [32, 64, 128],
            'lstm_hidden_size': [64, 128, 256],
            'num_layers': [1, 2, 3],
            'epochs': [100, 200, 300]
        }

        # Perform a grid search to find the optimal hyperparameters
        train_loader, val_loader, _ = self.prepare_data()
        best_params = self.grid_search(train_loader, val_loader, param_grid)

        # Train the model with the optimal hyperparameters using the whole dataset (train + validation)
        train_loader, _, _ = self.prepare_data(val_size=0)
        self.train(train_loader, None, epochs=best_params['epochs'], learning_rate=best_params['learning_rate'])

        # Evaluate the model on the test set
        self.evaluate(test_loader)


    def get_top_cryptos_to_buy(self, cryptos, n=2):
        self.model.eval()
        cryptos_data = []
        cryptos_labels = []

        # Get the transformed data for each cryptocurrency
        for crypto in cryptos:
            transformed_data = self.transformer.transform_crypto(crypto)
            X = transformed_data.drop(columns=['Returns']).values
            cryptos_data.append(torch.tensor(X, dtype=torch.float32))
            cryptos_labels.append(crypto)

        predicted_returns = []

        with torch.no_grad():
            for data in cryptos_data:
                data = data.permute(1, 0)
                data = data.unsqueeze(0)
                prediction = self.model(data)
                predicted_returns.append(prediction.item())

        # Sort the cryptocurrencies by their predicted returns in descending order
        sorted_cryptos = sorted(zip(cryptos_labels, predicted_returns), key=lambda x: x[1], reverse=True)

        # Return the top n cryptocurrencies
        return sorted_cryptos[:n]


    def retrain(self, load_data=True):
        print("Retraining the model...")
        if load_data:
            self.data.load_data()  # Add a method in the Data class to load new data
        train_loader, val_loader, test_loader = self.prepare_data()
        self.train(train_loader, val_loader)


    def retrain_periodically(self, interval, unit='hours'):
        # Set the interval to retrain the model
        if unit == 'minutes':
            schedule.every(interval).minutes.do(self.retrain)
        elif unit == 'hours':
            schedule.every(interval).hours.do(self.retrain)
        elif unit == 'days':
            schedule.every(interval).days.do(self.retrain)
        else:
            raise ValueError("Invalid unit for scheduling. Choose from 'minutes', 'hours', or 'days'.")

        # Run the scheduler
        print(f"Model will be retrained every {interval} {unit}.")
        while True:
            schedule.run_pending()
            time.sleep(60)


    def save_model(self, path='model.pth'):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


    def load_model(self, path='model.pth'):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            print(f"Model loaded from {path}")
        else:
            raise ValueError(f"No model found at {path}")

# # Load data
# data = Data()
# prices = data.load_data()

# # Preprocess data
# processor = Processor(prices)
# X, y, column_names = processor.process()

# # Perform EDA
# eda = EDA(X, y, column_names)

# # Transform data
# transformer = Transformer(eda)

# # Create CryptoModel and train it
# crypto_model = CryptoModel(data, eda, transformer)
# train_loader, val_loader, test_loader = crypto_model.prepare_data()
# crypto_model.train(train_loader, val_loader, epochs=100, learning_rate=0.001)

# # Evaluate the model
# crypto_model.evaluate(test_loader)
# crypto_model = CryptoModel(data, eda, transformer)
# crypto_model.optimize(test_loader)
# top_cryptos = crypto_model.get_top_cryptos_to_buy(cryptos)
# print("Top cryptocurrencies to buy:", top_cryptos)