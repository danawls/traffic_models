import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import glob


# 가중치 계산 함수
def calculate_weights(X, x_query, tau):
    """ 각 데이터 포인트에 대한 가중치를 계산합니다. """
    # 제곱 거리 계산
    distances = np.sum((X - x_query) ** 2, axis=1)
    # 가우시안 커널 가중치 계산
    weights = np.exp(-distances / (2 * tau ** 2))
    return np.diag(weights)


# Locally Weighted Linear Regression (LWL) 함수
def lwlr(X, y, x_query, tau):
    """ 주어진 쿼리 포인트에 대해 국소 가중치 회귀를 수행합니다. """
    # 가중치 행렬 계산
    W = calculate_weights(X, x_query, tau)
    # X에 상수 항 추가
    X_ = np.c_[np.ones(X.shape[0]), X]
    # 선형 회귀 파라미터 계산
    theta = np.linalg.pinv(X_.T @ W @ X_) @ X_.T @ W @ y
    # 예측값 계산
    x_query_ = np.r_[1, x_query]  # 상수 항 추가
    prediction = x_query_ @ theta
    return prediction


# 모델 학습 및 예측 함수
def train_lwlr_model(data, tau=0.5):
    """ 주어진 데이터에 대해 LWL 모델을 학습하고 예측합니다. """
    # 입력(X)과 출력(y) 정의
    X = data[['speed(u)', 'confusion']].values
    y = data['traffic(Q)'].values

    # 예측값 저장 리스트
    predictions = []

    # 각 데이터 포인트에 대해 LWL 수행
    for i in range(len(X)):
        prediction = lwlr(X, y, X[i], tau)
        predictions.append(prediction)

    return y, np.array(predictions)


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


# 데이터 로딩 및 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 날짜를 인덱스로 설정하고 정렬
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    # 필요한 컬럼 선택
    data = data[['traffic(Q)', 'speed(u)', 'confusion']]

    # 결측값이 있는 경우 처리 (여기서는 선형 보간 사용)
    data = data.interpolate(method='linear')

    return data


# 모든 CSV 파일을 불러오기 위한 경로 설정
file_paths = glob.glob('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/con/6000VDS02200.csv')

# 첫 번째 CSV 파일로 모델 학습 및 평가
if file_paths:  # 파일이 존재하는지 확인
    first_file = file_paths[0]
    data = preprocess_data(first_file)

    # Locally Weighted Linear Regression 모델 학습 및 예측
    tau = 0.5  # 가우시안 커널의 폭
    y_actual, y_pred = train_lwlr_model(data, tau)

    # 성능 평가
    evaluate_performance(y_actual, y_pred)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual, label='Actual Traffic(Q)', color='b')
    plt.plot(y_pred, label='Predicted Traffic(Q)', color='r', linestyle='--')
    plt.xlabel('Samples')
    plt.ylabel('Traffic(Q)')
    plt.title('Locally Weighted Linear Regression: Actual vs Predicted Traffic(Q)')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("지정된 디렉토리에 CSV 파일이 없습니다.")