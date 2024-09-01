import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob


# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 날짜를 인덱스로 설정하고 정렬
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    return data[['date', 'traffic(Q)']].rename(columns={'date': 'ds', 'traffic(Q)': 'y'})


# Prophet 모델 학습 및 예측 함수
def train_prophet_model(data):
    # 모델 초기화
    model = Prophet()

    # 모델 학습
    model.fit(data)

    # 미래 데이터프레임 생성 (학습 데이터의 기간만큼만 생성)
    future = data[['ds']]  # 학습 데이터의 날짜 범위로만 예측
    forecast = model.predict(future)

    return forecast


# 성능 평가 함수
def evaluate_performance(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted, squared=False)
    r2 = r2_score(actual, predicted)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")


# 모든 CSV 파일을 불러오기 위한 경로 설정
file_paths = glob.glob('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/con/6000VDS02200.csv')

# 첫 번째 CSV 파일로 모델 학습 및 평가
if file_paths:  # 파일이 존재하는지 확인
    first_file = file_paths[0]
    data = preprocess_data(first_file)

    # Prophet 모델 학습 및 예측
    forecast = train_prophet_model(data)

    # 실제 데이터와 예측 데이터의 길이를 맞추기 위해, 학습 데이터 기간만큼 자르기
    predicted = forecast['yhat'].values
    actual = data['y'].values

    # 성능 평가
    evaluate_performance(actual, predicted)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(data['ds'], actual, label='Actual Traffic(Q)', color='b')
    plt.plot(forecast['ds'], predicted, label='Predicted Traffic(Q)', color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Traffic(Q)')
    plt.title('Prophet: Actual vs Predicted Traffic(Q)')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("지정된 디렉토리에 CSV 파일이 없습니다.")