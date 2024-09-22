import pandas as pd
import numpy as np
from pydlm import dlm, trend, seasonality
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import glob

t = 10

# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 데이터 전처리: 날짜를 인덱스로 설정하고 정렬
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data = data.sort_index()

    return data


# DLM 모델 학습 및 예측 함수
def train_dlm_model(data):
    # 타겟 데이터 정의
    y = data['traffic(Q)'].values

    # 학습 데이터와 테스트 데이터 분할
    train_size = int(len(y) * 0.2)
    train, test = y[:train_size], y[:]

    # DLM 모델 정의: 여기서 추세(trend)와 계절성(seasonality)을 포함합니다
    my_dlm = dlm(train) + trend(degree=1, discount=0.9) + seasonality(period=32, discount=0.9)

    # 모델 학습
    my_dlm.fit()

    # 예측 수행
    predictions = my_dlm.predictN(date=my_dlm.n - 1, N=len(test))  # 수정된 부분

    # 예측 결과는 2차원 배열로 반환되므로, 이를 1차원으로 변환
    predictions = np.array(predictions).flatten()

    # 필요에 따라 예측 결과를 슬라이싱하여 테스트 데이터와 동일한 길이로 맞춤
    predictions = predictions[:len(test)]

    return test, predictions


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

    # DLM 모델 학습 및 예측
    y_test, y_pred = train_dlm_model(data)

    # 성능 평가
    evaluate_performance(y_test, y_pred)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(y_test):], y_test, label='Actual Traffic(Q)', color='b')
    plt.plot(data.index[-len(y_pred):], y_pred, label='Predicted Traffic(Q)', color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Traffic(Q)')
    plt.title('Dynamic Linear Model: Actual vs Predicted Traffic(Q)')
    plt.legend()
    plt.grid()
    plt.show()

    # df = pd.DataFrame({'value': y_pred, 'real': y_test})
    # df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/clock-compare/dlm.csv', index=False)
else:
    print("No CSV files found in the specified directory.")