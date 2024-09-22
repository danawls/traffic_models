import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import glob

t = 10

# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 날짜를 인덱스로 설정하고 정렬
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    # 필요한 컬럼 선택
    data = data[['date', 'traffic(Q)']]

    # 인덱스를 날짜로 설정
    data.set_index('date', inplace=True)

    # 결측값이 있는 경우 처리 (여기서는 선형 보간 사용)
    data = data.interpolate(method='linear')

    return data


# Holt-Winters 모델 학습 및 예측 함수
def train_holt_winters_model(data, seasonal_periods, forecast_steps=64):
    # Holt-Winters 모델 초기화
    model = ExponentialSmoothing(data['traffic(Q)'], trend='add', seasonal='add', seasonal_periods=seasonal_periods)

    # 모델 학습
    model_fit = model.fit()

    # 예측
    forecast = model_fit.forecast(steps=forecast_steps)

    return forecast


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
file_paths = glob.glob(f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/{t}/6000VDS03500.csv')

# 첫 번째 CSV 파일로 모델 학습 및 평가
if file_paths:  # 파일이 존재하는지 확인
    first_file = file_paths[0]
    data = preprocess_data(first_file)

    # Holt-Winters 모델 학습 및 예측
    seasonal_periods = 32  # 데이터 주기 (여기서는 예시로 12를 사용)
    forecast_steps = int(len(data) * 1)
    forecast = train_holt_winters_model(data, seasonal_periods, forecast_steps=forecast_steps)

    # 실제 값과 예측 값 비교
    actual = data['traffic(Q)'][-forecast_steps:].values
    predicted = forecast.values

    predicted = [-x for x in predicted]

    # 성능 평가
    evaluate_performance(actual, predicted)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Traffic(Q)', color='b')
    plt.plot(predicted, label='Predicted Traffic(Q)', color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Traffic(Q)')
    plt.title('Holt-Winters: Actual vs Predicted Traffic(Q)')
    plt.legend()
    plt.grid()
    plt.show()

    # df = pd.DataFrame({'value': predicted, 'real': actual})
    # df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/clock-compare/holt_windter.csv', index=False)
else:
    print("지정된 디렉토리에 CSV 파일이 없습니다.")