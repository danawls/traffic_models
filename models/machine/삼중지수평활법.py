import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import glob

t = 1

# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 데이터 전처리: 날짜를 인덱스로 설정하고 정렬
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data = data.sort_index()

    return data


# 삼중 지수 평활법 모델 학습 및 예측 함수
def train_holt_winters_model(data):
    # 타겟 데이터 정의
    y = data['traffic(Q)'].values

    # 학습 데이터와 테스트 데이터 분할
    train_size = int(len(y) * 0.2)
    train, test = y[:], y[:]

    # 삼중 지수 평활법 모델 학습
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=100)
    model_fit = model.fit(optimized=True)  # 최적의 매개변수를 자동으로 찾음

    # 테스트 데이터를 사용하여 예측 수행
    predictions = model_fit.forecast(len(test))

    return test, predictions


# 성능 평가 함수
def evaluate_performance(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# 모든 CSV 파일을 불러오기 위한 경로 설정
file_paths = glob.glob(f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/{t}/6000VDS03500.csv')

# 첫 번째 CSV 파일로 모델 학습 및 평가
if file_paths:  # 파일이 존재하는지 확인
    first_file = file_paths[0]
    data = preprocess_data(first_file)

    # 삼중 지수 평활법 모델 학습 및 예측
    y_test, y_pred = train_holt_winters_model(data)

    y_pred = [-x for x in y_pred]

    # 성능 평가
    evaluate_performance(y_test, y_pred)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(y_test):], y_test, label='Actual Traffic(Q)', color='b')
    plt.plot(data.index[-len(y_pred):], y_pred, label='Predicted Traffic(Q)', color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Traffic(Q)')
    plt.title('Holt-Winters Model: Actual vs Predicted Traffic(Q)')
    plt.legend()
    plt.grid()
    plt.show()

    df = pd.DataFrame({'value': y_pred, 'real': y_test})
    df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/clock-compare/three.csv', index=False)
else:
    print("No CSV files found in the specified directory.")