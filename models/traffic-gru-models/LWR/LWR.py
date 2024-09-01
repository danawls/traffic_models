import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob


# LWR 모델 함수 정의
def lwr_predictions(density, vf=120, kj=150):
    velocity = vf * (1 - density / kj)
    flow = density * velocity
    return density, velocity, flow


# 데이터 불러오기 및 전처리 함수
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # 데이터 전처리 과정 필요 시 추가
    return data


# GRU 모델 생성 함수
def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # 예측 출력 레이어
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# 모델 학습 및 파인 튜닝 함수
def train_and_fine_tune_model(data, model=None, pickle_path='lwr_gru_model.pkl'):
    x_t = data[['density', 'speed(u)', 'flow']].values  # 기존 입력 데이터

    # LWR 모델 예측 값 계산
    density_lwr, velocity_lwr, flow_lwr = lwr_predictions(data['density'].values)
    lwr_inputs = np.vstack([density_lwr, velocity_lwr, flow_lwr]).T

    # 기존 입력 데이터와 LWR 예측 데이터 결합
    x_prime_t = np.hstack([x_t, lwr_inputs])

    # 3차원 입력으로 변환 (samples, timesteps, features)
    x_prime_t = x_prime_t.reshape((x_prime_t.shape[0], 1, x_prime_t.shape[1]))

    y = data['target'].values  # 예측 대상 (예: 다음 시간 단계의 교통 상태)
    y = np.nan_to_num(y)  # NaN 값을 0으로 대체하여 처리

    if model is None:
        model = create_gru_model(input_shape=(x_prime_t.shape[1], x_prime_t.shape[2]))

    # 모델 학습
    history = model.fit(x_prime_t, y, epochs=100, batch_size=32, validation_split=0.2)

    # 모델 저장
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)

    return model, history


# 예측 및 성능 지표 출력 함수
def evaluate_model(model, data):
    x_t = data[['density', 'speed(u)', 'flow']].values  # 기존 입력 데이터

    # LWR 모델 예측 값 계산
    density_lwr, velocity_lwr, flow_lwr = lwr_predictions(data['density'].values)
    lwr_inputs = np.vstack([density_lwr, velocity_lwr, flow_lwr]).T

    # 기존 입력 데이터와 LWR 예측 데이터 결합
    x_prime_t = np.hstack([x_t, lwr_inputs])

    # 3차원 입력으로 변환 (samples, timesteps, features)
    x_prime_t = x_prime_t.reshape((x_prime_t.shape[0], 1, x_prime_t.shape[1]))

    predictions = model.predict(x_prime_t).flatten()  # 차원 축소

    # NaN 값이 있는 경우 제거
    valid_indices = ~np.isnan(data['target'])
    true_values = data['target'][valid_indices]
    predictions = predictions[valid_indices]

    # 성능 지표 계산
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R^2 Score: {r2:.4f}')

    plt.figure(figsize=(12, 6))
    plt.plot(data.index[valid_indices], true_values, label='True Values')
    plt.plot(data.index[valid_indices], predictions, label='Predicted Values', linestyle='--')
    plt.title('Traffic Flow Prediction using LWR-GRU Model')
    plt.xlabel('Time')
    plt.ylabel('Traffic Flow')
    plt.legend()
    plt.show()


def main():
    # 파일 리스트 생성
    file_paths = sorted(glob.glob('/Volumes/Expansion/traffic-prediction/product-data/con/6000VDS02200.csv'))


    # 첫 모델 학습
    model, history = None, None
    for file_path in file_paths:
        data = load_and_preprocess_data(file_path)
        # 교통 밀도, 속도, 흐름 계산
        data['density'] = data['traffic(Q)']  # 예시로 속도를 밀도로 사용
        data['flow'] = data['density'] * data['speed(u)']  # q_t = k_t * v_t
        data['target'] = data['speed(u)'].shift(-1)
        data.dropna(inplace=True)
        if model is None:
            model, history = train_and_fine_tune_model(data, model=None, pickle_path='lwr_gru_model.pkl')
        else:
            with open('lwr_gru_model.pkl', 'rb') as f:
                model = pickle.load(f)
            model, history = train_and_fine_tune_model(data, model=model, pickle_path='lwr_gru_model.pkl')

        # 예측 및 성능 지표 평가
        evaluate_model(model, data)


if __name__ == '__main__':
    main()