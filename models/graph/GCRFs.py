import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])  # 날짜 컬럼을 datetime으로 변환
    return data

# GCRF 그래프 생성
def create_gcrf_graph(time_steps):
    G = nx.path_graph(time_steps)  # 시간 기반 그래프 (시간 간의 상관관계 고려)
    return G

# 가우시안 조건부 랜덤 필드 (GCRF) 모델 훈련
def train_gcrf_model(data):
    # 타겟: traffic(Q) 열을 예측
    y = data['traffic(Q)'].values

    # 노드 생성 (시간에 따른 교통량을 노드로 정의)
    time_steps = len(y)
    gcrf_graph = create_gcrf_graph(time_steps)

    # 정밀도 행렬(Precision matrix) 설정
    precision = np.eye(time_steps) * 2  # 대각선은 자기 자신에 대한 상관관계
    for i in range(time_steps - 1):
        precision[i, i+1] = -1  # 인접한 시간의 노드와 상관관계 설정
        precision[i+1, i] = -1

    # 선형 회귀 기반 잠정 예측 값 (Baseline predictions using linear regression)
    X = np.arange(time_steps).reshape(-1, 1)  # 시간 단계
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X, y)
    y_pred_baseline = reg.predict(X)

    # GCRF를 이용한 예측
    y_pred_gcrf = np.zeros(time_steps)
    for i in range(1, time_steps - 1):
        # 현재 상태는 이웃 노드들에 의존 (가우시안 조건부 랜덤 필드의 특성)
        y_pred_gcrf[i] = 0.5 * (y_pred_baseline[i-1] + y_pred_baseline[i+1])

    return y, y_pred_gcrf

# 성능 평가 함수
def evaluate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R^2: {r2}")
    print(f"MAPE: {mape}")

    return mse, rmse, mae, r2

# 시각화 함수
def visualize_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(data['traffic(Q)'], label="Actual Traffic (Q)")
    plt.plot(y_pred, label="Predicted Traffic (Q) - GCRF", linestyle='--')
    plt.xlabel("Time Steps")
    plt.ylabel("Traffic (Q)")
    plt.legend()
    plt.title("GCRF Traffic Prediction")
    plt.show()

# 실행 코드
file_path = f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/10/6000VDS03500.csv'  # 파일 경로를 지정하세요
data = preprocess_data(file_path)

# 모델 학습 및 예측
y_actual, y_predicted_gcrf = train_gcrf_model(data)

# 성능 평가
evaluate_performance(y_actual, y_predicted_gcrf)

# 결과 시각화
visualize_results(y_actual, y_predicted_gcrf)

# df = pd.DataFrame({'value': list(y_predicted_gcrf), 'real': data['traffic(Q)']})
# df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/graph-compare/mcrfs.csv', index=False)