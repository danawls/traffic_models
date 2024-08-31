import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Concatenate
import matplotlib.pyplot as plt
import glob

K = 1.5

# 시계열 데이터 생성 함수
def create_sequences(data, shock_data, seq_length):
    x = []
    y = []
    shock = []
    for i in range(len(data)-seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
        shock.append(shock_data[i+seq_length])  # 충격파 강도 추가
    return np.array(x), np.array(y), np.array(shock)

# CSV 파일들을 로드하고 순차적으로 모델 학습 및 파인튜닝
file_list = sorted(glob.glob('/Volumes/Expansion/traffic-prediction/product-data/32.csv'))  # '1.csv', '2.csv', '3.csv' 등
seq_length = 10

# 모델 초기화 변수
model = None
pickle_path = 'moving_average_gru_model.pkl'

# CSV 파일 순회
for i, file_path in enumerate(file_list):
    # 1. 데이터 준비 및 전처리
    # 데이터 로드
    data = pd.read_csv(file_path)

    # 예를 들어 'Volume' 열이 교통량 데이터라고 가정
    # 이동 평균 계산 (예: 5일 이동 평균)
    data['Moving_Avg'] = data['통행속도'].rolling(window=5).mean()

    # 이동 평균과의 차이를 계산하여 충격파 강도 정의
    data['Volume_Change'] = data['통행속도'] - data['Moving_Avg']

    # 표준편차를 이용한 임계값 설정 (예: 이동 평균 변화의 1.5배 표준편차를 충격파로 간주)
    threshold = data['Volume_Change'].std() * K
    data['Shock_Intensity'] = np.where(data['Volume_Change'].abs() > threshold, 1, 0)

    # NaN 값 제거 (이동 평균 계산으로 인해 생길 수 있는 NaN 값)
    data = data.dropna()

    # 데이터 정규화 (Min-Max Scaler)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['통행속도']])
    shock_intensity = data['Shock_Intensity'].values  # 충격파 강도 배열로 변환
    X, y, shock_intensity_seq = create_sequences(data_scaled, shock_intensity, seq_length)

    # 데이터 나누기 (Train/Test Split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    shock_train, shock_test = shock_intensity_seq[:train_size], shock_intensity_seq[train_size:]

    # 충격파 강도의 형상 조정
    shock_train = shock_train.reshape(-1, 1)
    shock_test = shock_test.reshape(-1, 1)

    # 2. 모델 구성 및 학습
    if i == 0:
        # 초기 모델 구성 (첫 번째 파일만)
        input_gru = Input(shape=(X_train.shape[1], X_train.shape[2]))
        shock_intensity_input = Input(shape=(1,))  # 충격파 강도 입력

        # GRU Layer
        gru_out = GRU(64, return_sequences=False)(input_gru)

        # 충격파 강도를 GRU 결과와 결합
        concat = Concatenate()([gru_out, shock_intensity_input])
        dense_out = Dense(32, activation='relu')(concat)
        gru_output = Dense(1)(dense_out)

        model = Model(inputs=[input_gru, shock_intensity_input], outputs=gru_output)

        # GRU 모델 컴파일
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    else:
        # 기존 모델 불러오기 및 파인튜닝
        with open(pickle_path, 'rb') as f:
            model, scaler = pickle.load(f)

    # 3. 모델 학습 또는 파인튜닝
    model.fit([X_train, shock_train], y_train, epochs=500, batch_size=32, validation_split=0.2)

    # 4. 모델 평가
    y_train_pred = model.predict([X_train, shock_train])
    y_test_pred = model.predict([X_test, shock_test])

    # 성능 지표 계산
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Train MSE: {train_mse}, Train MAE: {train_mae}, Train R2: {train_r2}")
    print(f"Test MSE: {test_mse}, Test MAE: {test_mae}, Test R2: {test_r2}")

    # 정규화 해제
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    train_predictions = scaler.inverse_transform(y_train_pred).flatten()
    test_predictions = scaler.inverse_transform(y_test_pred).flatten()

    # 5. 예측 결과 시각화
    look_back = seq_length  # 시퀀스 길이만큼 이동
    plt.figure(figsize=(14, 5))

    # 학습 데이터 시각화
    plt.plot(data['date'].iloc[look_back:train_size + look_back], y_train_inv[:train_size], label='Train Actual')
    plt.plot(data['date'].iloc[look_back:train_size + look_back], train_predictions[:train_size], label='Train Predictions')

    # 테스트 데이터 시각화
    plt.plot(data['date'].iloc[train_size + look_back:train_size + look_back + len(y_test_inv)], y_test_inv[:len(data['date'].iloc[train_size + look_back:train_size + look_back + len(y_test_inv)])], label='Test Actual')
    plt.plot(data['date'].iloc[train_size + look_back:train_size + look_back + len(test_predictions)], test_predictions[:len(data['date'].iloc[train_size + look_back:train_size + look_back + len(test_predictions)])], label='Test Predictions')

    plt.title(f'GRU Model Predictions vs Actual (File: {file_path})')
    plt.xlabel('Date')
    plt.ylabel('Flow')
    plt.xticks(np.arange(0, 6072, 200), rotation=90, size=10)
    plt.legend()
    plt.show()

    # 모델 저장 (업데이트)
    with open(pickle_path, 'wb') as f:
        pickle.dump((model, scaler), f)