import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the data
file_path = 'BPCL.csv'  # Change this to your file path
data = pd.read_csv(file_path)

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

# Build and Train LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

# Make Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
Y_train_inv = scaler.inverse_transform(Y_train.reshape(-1, 1))
Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Calculate performance metrics
train_mse = mean_squared_error(Y_train_inv, train_predict)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(Y_train_inv, train_predict)
train_r2 = r2_score(Y_train_inv, train_predict)

test_mse = mean_squared_error(Y_test_inv, test_predict)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(Y_test_inv, test_predict)
test_r2 = r2_score(Y_test_inv, test_predict)

# Output performance metrics
print(f"Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train MAE: {train_mae}, Train R2: {train_r2}")
print(f"Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test R2: {test_r2}")

# Plot training & validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/figure/gru_compare/lstm.png', dpi=500)
plt.show()