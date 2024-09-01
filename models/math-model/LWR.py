import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob


# LWR 모델 파라미터 설정 및 초기화
def compute_lwr_parameters(data):
    V_max = data['speed(u)'].max()  # 데이터에서 최대 속도 (m/s)
    rho_max = data['traffic(Q)'].max() / data['lane_number'].max()  # 데이터에서 최대 교통량을 밀도로 사용
    return V_max, rho_max


# LWR 모델 흐름 계산 함수
def compute_lwr_flow(density, V_max, rho_max):
    # 흐름-밀도 관계에 따른 흐름 계산
    return density * V_max * (1 - density / rho_max)


# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 데이터 전처리
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data = data.sort_index()

    # 이상치 제거
    data = data[(data['traffic(Q)'] > 0) & (data['traffic(Q)'] < data['traffic(Q)'].quantile(0.99))]
    data = data[(data['speed(u)'] > 0) & (data['speed(u)'] < data['speed(u)'].quantile(0.99))]

    # 밀도 계산: traffic(Q) / lane_number
    data['density'] = data['traffic(Q)'] / data['lane_number']

    return data


# LWR 모델을 사용하여 다음 교통량을 예측하는 함수
def predict_traffic(data, V_max, rho_max, dx=0.1, dt=0.01, T=10):
    # 초기 밀도 설정
    rho = data['density'].values

    # 시뮬레이션을 위한 시간 루프
    timesteps = int(T / dt)
    traffic_result = [data['traffic(Q)'].values]  # 초기 교통량 저장

    for t in range(timesteps):
        # 흐름 계산
        q = compute_lwr_flow(rho, V_max, rho_max)

        # 공간에 따른 차분 계산
        q_plus = np.roll(q, -1)  # 다음 위치의 흐름 (앞으로 이동)
        dqdx = (q_plus - q) / dx

        # 밀도 갱신: 연속 방정식을 사용하여 밀도 변화 반영
        rho = rho - dt * dqdx

        # NaN 또는 무한대 값이 발생하지 않도록 클리핑
        rho = np.clip(rho, 0, rho_max)

        # 새로운 밀도를 사용하여 교통량 Q 계산
        traffic = rho * data['lane_number'].max()  # lane_number로 다시 교통량으로 변환
        traffic_result.append(traffic)

    return np.array(traffic_result)


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

# 첫 번째 CSV 파일로 모델 초기화
first_file = file_paths[0]
data = preprocess_data(first_file)
V_max, rho_max = compute_lwr_parameters(data)

# LWR 모델을 사용하여 다음 교통량 예측
traffic_result = predict_traffic(data, V_max, rho_max)

# 성능 평가
actual_traffic = data['traffic(Q)'].values
predicted_traffic = traffic_result[-1]  # 마지막 시간 스텝의 예측 교통량
evaluate_performance(actual_traffic, predicted_traffic)

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(data.index, actual_traffic, label='Actual Traffic(Q)', color='b')
plt.plot(data.index, predicted_traffic, label='Predicted Traffic(Q)', color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Traffic(Q)')
plt.title('LWR Model: Actual vs Predicted Traffic(Q)')
plt.legend()
plt.grid()
plt.show()