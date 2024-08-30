import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GRU, Concatenate

K = 1.5

# 1. 데이터 준비 및 전처리
# 데이터 로드
data = pd.read_csv('BPCL.csv')

# 예를 들어 'Volume' 열이 교통량 데이터라고 가정
# 이동 평균 계산 (예: 5일 이동 평균)
data['Moving_Avg'] = data['Volume'].rolling(window=5).mean()

# 이동 평균과의 차이를 계산하여 충격파 강도 정의
data['Volume_Change'] = data['Volume'] - data['Moving_Avg']

# 표준편차를 이용한 임계값 설정 (예: 이동 평균 변화의 1.5배 표준편차를 충격파로 간주)
threshold = data['Volume_Change'].std() * K
data['Shock_Intensity'] = np.where(data['Volume_Change'].abs() > threshold, 1, 0)

# NaN 값 제거 (이동 평균 계산으로 인해 생길 수 있는 NaN 값)
data = data.dropna()

# 데이터 정규화 (Min-Max Scaler)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Volume']])
shock_intensity = data['Shock_Intensity'].values  # 충격파 강도 배열로 변환

# 시계열 데이터 생성
def create_sequences(data, shock_data, seq_length):
    x = []
    y = []
    shock = []
    for i in range(len(data)-seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
        shock.append(shock_data[i+seq_length])  # 충격파 강도 추가
    return np.array(x), np.array(y), np.array(shock)

seq_length = 10
X, y, shock_intensity_seq = create_sequences(data_scaled, shock_intensity, seq_length)

# 데이터 나누기 (Train/Test Split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
shock_train, shock_test = shock_intensity_seq[:train_size], shock_intensity_seq[train_size:]

# 2. 모델 구성
# GRU 모델 구성
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

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# 3. 모델 학습
history = model.fit([X_train, shock_train], y_train, epochs=50, batch_size=32, validation_split=0.2)

# 4. 모델 평가
# 학습 데이터 예측
y_train_pred = model.predict([X_train, shock_train])
# 테스트 데이터 예측
y_test_pred = model.predict([X_test, shock_test])

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