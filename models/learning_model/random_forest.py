import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob


# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 데이터 전처리
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data = data.sort_index()


    return data


# 랜덤 포레스트 회귀 모델 학습 및 예측
def train_random_forest_model(data):
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

    # 랜덤 포레스트 모델 학습
    rf_model = RandomForestRegressor(n_estimators=1, random_state=42)
    rf_model.fit(X_train, y_train)

    # 예측 수행
    y_pred_scaled = rf_model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()  # 스케일링 되돌리기
    y_test_original = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()  # 스케일링 되돌리기

    return y_test_original, y_pred


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
first_file = file_paths[0]
data = preprocess_data(first_file)

# 랜덤 포레스트 모델 학습 및 예측
y_test, y_pred = train_random_forest_model(data)

# 성능 평가
evaluate_performance(y_test, y_pred)

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Traffic(Q)', color='b')
plt.plot(y_pred, label='Predicted Traffic(Q)', color='r', linestyle='--')
plt.xlabel('Sample')
plt.ylabel('Traffic(Q)')
plt.title('Random Forest Model: Actual vs Predicted Traffic(Q)')
plt.legend()
plt.grid()
plt.show()

# df = pd.DataFrame({'value': list(y_pred)})
# df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/deep-compare/random_forest.csv', index=False)