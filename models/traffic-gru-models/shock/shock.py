import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GRU, Concatenate

# 1. 데이터 준비
data = pd.read_csv('BPCL.csv')
# 데이터 확인 및 필요한 전처리
print(data.head())

# 2. 데이터 전처리
# 예시: Close 가격을 예측한다고 가정
# 데이터 정규화 (Min-Max Scaler)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close']])

# 시계열 데이터 생성
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 10
X, y = create_sequences(data_scaled, seq_length)

# 데이터 나누기 (Train/Test Split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. 모델 구성
# 충격파 강도 예측 ANN 모델
input_ann = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(input_ann)
x = Dense(32, activation='relu')(x)
shock_intensity_output = Dense(1, activation='linear')(x)
ann_model = Model(inputs=input_ann, outputs=shock_intensity_output)

# GRU 모델
input_gru = Input(shape=(X_train.shape[1], X_train.shape[2]))
shock_intensity_input = Input(shape=(1,))  # 충격파 강도 입력
gru_out = GRU(64, return_sequences=False)(input_gru)
concat = Concatenate()([gru_out, shock_intensity_input])
dense_out = Dense(32, activation='relu')(concat)
gru_output = Dense(1)(dense_out)

# 최종 모델
model = Model(inputs=[input_gru, shock_intensity_input], outputs=gru_output)

# 4. 모델 학습
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
history = model.fit([X_train, y_train], y_train, epochs=100, batch_size=32, validation_split=0.2)

# 5. 모델 평가
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_train_pred = model.predict([X_train, y_train])
y_test_pred = model.predict([X_test, y_test])

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train MAE: {train_mae}, Train R2: {train_r2}")
print(f"Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test R2: {test_r2}")

# 6. 예측 수행
# 새로운 데이터를 예측하려면 모델에 새로운 입력을 제공하면 됩니다.