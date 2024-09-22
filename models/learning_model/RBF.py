import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import glob


# 데이터 로딩 및 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 날짜를 인덱스로 설정하고 정렬
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    # 필요한 컬럼 선택
    data = data[['traffic(Q)', 'speed(u)', 'confusion', 'lane_number']]

    # 결측값이 있는 경우 처리 (여기서는 선형 보간 사용)
    data = data.interpolate(method='linear')

    return data


# RBF 신경망 모델 학습 및 예측 함수
def train_rbf_model(data):
    # 입력(X)과 출력(y) 정의
    X = data[['speed(u)', 'confusion', 'lane_number']].values
    y = data['traffic(Q)'].values

    # 데이터 정규화
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 훈련 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # RBF 신경망 구성
    rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=10)
    ridge_reg = Ridge(alpha=1.0)
    model = make_pipeline(rbf_feature, ridge_reg)

    # 모델 학습
    model.fit(X_train, y_train)

    # 예측
    y_pred_scaled = model.predict(X_scaled)
    y_test = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    return y_test, y_pred


# 성능 평가 함수
def evaluate_performance(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# 모든 CSV 파일을 불러오기 위한 경로 설정
file_paths = glob.glob(f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/10/6000VDS03500.csv')

# 첫 번째 CSV 파일로 모델 학습 및 평가
if file_paths:  # 파일이 존재하는지 확인
    first_file = file_paths[0]
    data = preprocess_data(first_file)

    # RBF 모델 학습 및 예측
    y_test, y_pred = train_rbf_model(data)

    # 성능 평가
    evaluate_performance(y_test, y_pred)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Traffic(Q)', color='b')
    plt.plot(y_pred, label='Predicted Traffic(Q)', color='r', linestyle='--')
    plt.xlabel('Samples')
    plt.ylabel('Traffic(Q)')
    plt.title('RBF Neural Network: Actual vs Predicted Traffic(Q)')
    plt.legend()
    plt.grid()
    plt.show()

    # df = pd.DataFrame({'value': list(y_pred), 'real': list(y_test)})
    # df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/deep-compare/rbf.csv', index=False)
else:
    print("지정된 디렉토리에 CSV 파일이 없습니다.")