import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import glob


# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 데이터 전처리
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data = data.sort_index()

    # 이상치 제거 및 필요한 컬럼만 유지
    data = data[(data['traffic(Q)'] > 0) & (data['traffic(Q)'] < data['traffic(Q)'].quantile(0.99))]
    data = data[(data['speed(u)'] > 0) & (data['speed(u)'] < data['speed(u)'].quantile(0.99))]

    return data


# LSTM 모델 정의
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='mse')

    return model


# LSTM 모델을 사용하여 교통량 예측
def train_lstm_model(data):
    # 입력 데이터와 타겟 데이터 정의
    X = data[['speed(u)', 'confusion', 'lane_number']].values
    y = data['traffic(Q)'].shift(-1).ffill().values  # 다음 시간대의 교통량을 타겟으로 설정

    # 데이터 스케일링
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 학습 데이터와 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # LSTM은 3D 입력이 필요하므로 데이터 재구성
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # LSTM 모델 학습
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = build_lstm_model(input_shape)
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # 예측 수행
    y_pred_scaled = lstm_model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()  # 스케일링 되돌리기
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()  # 스케일링 되돌리기

    return y_test_original, y_pred


# 성능 평가 함수
def evaluate_performance(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")


# 모든 CSV 파일을 불러오기 위한 경로 설정
file_paths = glob.glob('/Volumes/Expansion/traffic-prediction/product-data/con/6000VDS02200.csv')

# 첫 번째 CSV 파일로 모델 학습 및 평가
first_file = file_paths[0]
data = preprocess_data(first_file)

# LSTM 모델 학습 및 예측
y_test, y_pred = train_lstm_model(data)

# 성능 평가
evaluate_performance(y_test, y_pred)

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Traffic(Q)', color='b')
plt.plot(y_pred, label='Predicted Traffic(Q)', color='r', linestyle='--')
plt.xlabel('Sample')
plt.ylabel('Traffic(Q)')
plt.title('LSTM Model: Actual vs Predicted Traffic(Q)')
plt.legend()
plt.grid()
plt.show()