import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Concatenate
import matplotlib.pyplot as plt

# 1. 데이터 준비 및 전처리
# 데이터 로드 (예: CSV 파일에서 교통량 데이터 로드)
data = pd.read_csv('1.csv')

# 예를 들어 'Volume' 열이 교통량 데이터라고 가정
data['Volume_Change'] = data['통행속도'].diff()  # 교통량 변화율 계산

# NaN 값 제거 (변화율 계산으로 인해 생길 수 있는 NaN 값)
data = data.dropna()

# 데이터 정규화 (Min-Max Scaler)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['통행속도']])

# 충격파 강도를 예측하기 위한 추가 입력 데이터 생성
shock_input_data = data_scaled  # 충격파 강도 예측을 위한 입력 데이터

# 시계열 데이터 생성
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

seq_length = 10
X, y = create_sequences(data_scaled, seq_length)
shock_X, shock_y = create_sequences(shock_input_data, seq_length)

# 데이터 나누기 (Train/Test Split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
shock_X_train, shock_X_test = shock_X[:train_size], shock_X[train_size:]
shock_y_train, shock_y_test = shock_y[:train_size], shock_y[train_size:]

# 2. 충격파 강도 예측을 위한 ANN 모델 구성
input_ann = Input(shape=(shock_X_train.shape[1], shock_X_train.shape[2]))
ann_out = Dense(64, activation='relu')(input_ann)
ann_out = Dense(32, activation='relu')(ann_out)
shock_intensity_output = Dense(1, activation='linear')(ann_out)
ann_model = Model(inputs=input_ann, outputs=shock_intensity_output)

# ANN 모델 컴파일
ann_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# ANN 모델 학습
ann_model.fit(shock_X_train, shock_y_train, epochs=50, batch_size=32, validation_split=0.2)

# 충격파 강도 예측
shock_train_pred = ann_model.predict(shock_X_train)
shock_test_pred = ann_model.predict(shock_X_test)

# 3. GRU 모델 구성
input_gru = Input(shape=(X_train.shape[1], X_train.shape[2]))
shock_intensity_input = Input(shape=(1,))  # 충격파 강도 입력

# GRU Layer
gru_out = GRU(64, return_sequences=False)(input_gru)

# 충격파 강도를 GRU 결과와 결합
concat = Concatenate()([gru_out, shock_intensity_input])
dense_out = Dense(32, activation='relu')(concat)
gru_output = Dense(1)(dense_out)

# 최종 모델 정의
model = Model(inputs=[input_gru, shock_intensity_input], outputs=gru_output)

# GRU 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# 4. GRU 모델 학습
history = model.fit([X_train, shock_train_pred], y_train, epochs=50, batch_size=32, validation_split=0.2)

# 5. 모델 평가
# 학습 데이터 예측
y_train_pred = model.predict([X_train, shock_train_pred])
# 테스트 데이터 예측
y_test_pred = model.predict([X_test, shock_test_pred])

# 성능 평가
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

# 6. 예측 결과 시각화
# 학습 데이터 예측 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_train, label='Actual Train Data')
plt.plot(y_train_pred, label='Predicted Train Data')
plt.title('Train Data: Actual vs Predicted')
plt.xlabel('Samples')
plt.ylabel('Scaled Volume')
plt.legend()
plt.show()

# 테스트 데이터 예측 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Test Data')
plt.plot(y_test_pred, label='Predicted Test Data')
plt.title('Test Data: Actual vs Predicted')
plt.xlabel('Samples')
plt.ylabel('Scaled Volume')
plt.legend()
plt.show()