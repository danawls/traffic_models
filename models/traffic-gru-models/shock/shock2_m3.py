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
from keras.models import Sequential

# 모델의 모든 정보를 지정해주신 형식으로 작성해 보겠습니다.
#
# ### 모델 정보
#
# 1. **모델의 유닛**
#    - GRU 레이어 유닛 수: 64개
#    - Dense 레이어 유닛 수: 32개 (첫 번째 Dense 레이어)
#    - 최종 출력 유닛 수: 1개 (최종 Dense 레이어)
#
# 2. **입력 데이터**
#    - `X_train`, `X_test`: 교통량 데이터의 시계열 입력으로, 각 시퀀스의 길이는 `seq_length`(예: 10)로 설정됩니다.
#    - `shock_train`, `shock_test`: 충격파 강도 데이터로, 시퀀스의 마지막 충격파 강도를 사용하여 GRU 출력과 결합합니다.
#
# 3. **데이터 전처리**
#    - `MinMaxScaler`를 사용하여 교통량 데이터를 0과 1 사이로 정규화합니다.
#    - 교통량의 변화율을 계산하고, 변화율의 표준편차를 기반으로 충격파 강도를 정의합니다.
#
# 4. **모델 구성**
#    - **입력 레이어**: GRU 레이어에 들어가는 시계열 입력(`Input(shape=(X_train.shape[1], X_train.shape[2]))`)과 충격파 강도 입력(`Input(shape=(1,))`).
#    - **GRU 레이어**: 64개의 유닛을 사용하여 시계열 데이터를 처리하고, 시간 의존적인 특성을 학습합니다.
#    - **Concatenate 레이어**: GRU 출력과 충격파 강도를 결합합니다.
#    - **Dense 레이어**: 32개의 유닛을 가진 첫 번째 Dense 레이어와 1개의 유닛을 가진 최종 출력 Dense 레이어로 구성되어 있습니다.
#
# 5. **모델 컴파일**
#    - 옵티마이저: Adam 옵티마이저 사용 (`learning_rate=0.001`)
#    - 손실 함수: Mean Squared Error (MSE)
#
# 6. **훈련 설정**
#    - Epochs: 50
#    - Batch Size: 32
#    - Validation Split: 0.2
#
# 7. **모델 저장 및 불러오기**
#    - 학습이 끝난 후 모델과 스케일러를 피클 파일로 저장하여 나중에 불러와 파인튜닝에 사용합니다.
#    - 피클 파일 경로: `'standard_deviation_gru_model.pkl'`
#
# 8. **모델 평가 지표**
#    - 학습 후, Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R2) 등의 지표를 사용하여 모델의 성능을 평가합니다.
#
# 9. **시각화**
#    - 학습 및 테스트 데이터에 대한 실제값과 예측값을 날짜별로 시각화합니다.
#    - `matplotlib` 라이브러리를 사용하여 시각화합니다.
#
# 이렇게 모델의 주요 구성 요소와 그 세부 사항을 정리해 보았습니다. 각 항목은 모델의 아키텍처, 데이터 처리 방식, 훈련 설정, 평가 방법, 시각화 등에 대한 정보를 포함하고 있습니다.

# 시계열 데이터 생성 함수
def create_sequences(data, shock_data, seq_length):
    x = []
    y = []
    shock = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
        shock.append(shock_data[i + seq_length - 1])  # 시퀀스의 마지막 충격파 강도만 고려
    return np.array(x), np.array(y), np.array(shock)

# CSV 파일들을 로드하고 순차적으로 모델 학습 및 파인튜닝
file_list = sorted(glob.glob('/Volumes/Expansion/traffic-prediction/product-data/32.csv'))  # '1.csv', '2.csv', '3.csv' 등
# file_list = sorted(glob.glob('archive/*.csv'))  # '1.csv', '2.csv', '3.csv' 등
seq_length = 10

# 모델 초기화 변수
model = None
pickle_path = 'standard_deviation_gru_model.pkl'

# CSV 파일 순회
for i, file_path in enumerate(file_list):
    # 1. 데이터 준비 및 전처리
    data = pd.read_csv(file_path)

    # '통행속도' 열이 교통량 데이터라고 가정하고, 'date' 열이 존재한다고 가정
    data = data.dropna()

    # 데이터 정규화 (Min-Max Scaler)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['통행속도']])  # '통행속도' 열 사용

    # 교통량 변화율 계산
    data['Volume_Change'] = data['통행속도'].diff()

    # NaN 값 제거 (변화율 계산으로 인해 생길 수 있는 NaN 값)
    data = data.dropna()

    # 표준편차 기반 충격파 강도 계산
    threshold = data['Volume_Change'].std()
    data['Shock_Intensity'] = np.where(data['Volume_Change'].abs() > threshold, 1, 0)

    # 시계열 데이터 생성
    shock_intensity = data['Shock_Intensity'].values.reshape(-1, 1)
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

        # 모델 저장
        with open(pickle_path, 'wb') as f:
            pickle.dump((model, scaler), f)
    else:
        # 기존 모델 불러오기 및 파인튜닝
        with open(pickle_path, 'rb') as f:
            model, scaler = pickle.load(f)

    # 3. 모델 학습 또는 파인튜닝
    model.fit([X_train, shock_train], y_train, epochs=100, batch_size=32, validation_split=0.2)

    # 4. 모델 평가
    y_train_pred = model.predict([X_train, shock_train])
    y_test_pred = model.predict([X_test, shock_test])

    # 정규화 해제
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    train_predictions = scaler.inverse_transform(y_train_pred).flatten()
    test_predictions = scaler.inverse_transform(y_test_pred).flatten()

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

    # 5. 예측 결과 시각화
    look_back = seq_length  # 시퀀스 길이만큼 이동
    plt.figure(figsize=(14, 5))

    # 학습 데이터 시각화
    # plt.figure(dpi=1000)

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