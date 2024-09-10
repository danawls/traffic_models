import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import inv

# 데이터 전처리 함수 정의
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()  # 결측치 제거
    return data

# GMRF 회귀 모델 학습 및 예측 함수 정의
def train_mrf_model(data):
    # 입력 데이터 및 타겟 데이터 설정
    X = data[['speed(u)', 'confusion', 'lane_number']].values
    y = data['traffic(Q)'].shift(-1).ffill().values  # 다음 시간대의 교통량을 타겟으로 설정

    # 전체 데이터에 대한 GMRF 설정
    n = X.shape[0]
    Q = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).tocsc()  # 희소 라플라시안 행렬 설정
    sigma2 = 1.0  # 잡음 분산
    Lambda = inv(Q + eye(n) / sigma2)  # 공분산 행렬

    # 전체 데이터에 대해 예측 수행
    y_pred = Lambda @ y

    return y, y_pred

# 성능 평가 함수 정의
def evaluate_performance(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}')

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted', linestyle='--')
    plt.xlabel('Sample')
    plt.ylabel('Traffic (Q)')
    plt.title('GMRF Regression Performance')
    plt.legend()
    plt.show()

# 메인 코드 실행
if __name__ == "__main__":
    # 데이터 파일 경로 설정
    file_path = '/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/con/6000VDS02200.csv'  # 여기에 사용자의 데이터 파일 경로를 입력하세요
    data = preprocess_data(file_path)

    # 모델 학습 및 예측
    y_actual, y_pred = train_mrf_model(data)

    # 성능 평가
    evaluate_performance(y_actual, y_pred)