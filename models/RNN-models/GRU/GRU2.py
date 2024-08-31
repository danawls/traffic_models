import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('1.csv')

# Preprocess the data
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')
prices = data['통행속도'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X, Y = create_dataset(scaled_prices, look_back)

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the GRU model
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(GRU(units=50, return_sequences=False))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the predictions
train_predictions = scaler.inverse_transform(train_predictions)
Y_train_inv = scaler.inverse_transform(Y_train.reshape(-1, 1))
test_predictions = scaler.inverse_transform(test_predictions)
Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Calculate performance metrics
train_mse = mean_squared_error(Y_train_inv, train_predictions)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(Y_train_inv, train_predictions)
train_r2 = r2_score(Y_train_inv, train_predictions)

test_mse = mean_squared_error(Y_test_inv, test_predictions)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(Y_test_inv, test_predictions)
test_r2 = r2_score(Y_test_inv, test_predictions)

print(f'Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train MAE: {train_mae}, Train R2: {train_r2}')
print(f'Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test R2: {test_r2}')

# Visualization
plt.figure(figsize=(14, 5))
plt.plot(data['date'][look_back:train_size + look_back], Y_train_inv, label='Train Actual')
plt.plot(data['date'][look_back:train_size + look_back], train_predictions, label='Train Predictions')
plt.plot(data['date'][train_size + look_back:], Y_test_inv, label='Test Actual')
plt.plot(data['date'][train_size + look_back:], test_predictions, label='Test Predictions')
plt.title('GRU Model Predictions vs Actual')
plt.xlabel('Date')
plt.ylabel('Flow')
plt.legend()
plt.show()