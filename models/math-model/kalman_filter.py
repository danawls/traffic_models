import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 칼만 필터 클래스 정의
class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x):
        self.A = A  # 상태 전이 행렬 (State transition matrix)
        self.B = B  # 제어 입력 행렬 (Control input matrix)
        self.H = H  # 관측 모델 행렬 (Observation model matrix)
        self.Q = Q  # 과정 잡음 공분산 (Process noise covariance)
        self.R = R  # 관측 잡음 공분산 (Measurement noise covariance)
        self.P = P  # 추정 공분산 (Estimation covariance)
        self.x = x  # 상태 변수 초기값 (Initial state)

    def predict(self, u=0):
        # 상태 예측
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        # 추정 공분산 예측
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        # 칼만 이득 계산
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # 상태 업데이트
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        # 추정 공분산 업데이트
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x

# 데이터 불러오기 (사용자 데이터 경로로 변경)
data = pd.read_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/con/6000VDS02200.csv')

# 'traffic(Q)' 열을 시계열 데이터로 사용
measurements = data['traffic(Q)'].values

# 칼만 필터 초기값 설정
dt = 1.0  # 시간 간격
A = np.array([[1, dt], [0, 1]])  # 상태 전이 행렬
B = np.array([[0.5 * dt**2], [dt]])  # 제어 입력 행렬
H = np.array([[1, 0]])  # 관측 모델 행렬
Q = np.array([[1, 0], [0, 1]]) * 0.001  # 과정 잡음 공분산
R = np.array([[1]])  # 관측 잡음 공분산
P = np.eye(2)  # 추정 공분산 초기값
x = np.array([[0], [0]])  # 상태 변수 초기값

# 칼만 필터 객체 생성
kf = KalmanFilter(A, B, H, Q, R, P, x)

# 칼만 필터를 통한 예측 및 업데이트
predictions = []
for z in measurements:
    pred = kf.predict()
    predictions.append(pred[0, 0])
    kf.update(np.array([[z]]))

mse = mean_squared_error(measurements, predictions)
mae = mean_absolute_error(measurements, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(measurements, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# 실제 값과 예측 값 시각화
plt.figure(figsize=(12, 6))
plt.plot(measurements, label='Actual Traffic (Q)')
plt.plot(predictions, label='Kalman Filter Prediction')
plt.xlabel('Time')
plt.ylabel('Traffic (Q)')
plt.title('Traffic Prediction using Kalman Filter')
plt.legend()
plt.show()