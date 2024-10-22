import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import glob

t = 10

# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 날짜를 인덱스로 설정하고 정렬
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    return data[['date', 'traffic(Q)']]


# GARCH 모델 학습 및 예측 함수
def train_garch_model(data):
    # 데이터 준비
    y = data['traffic(Q)'].values

    # GARCH(1, 1) 모델 초기화
    model = arch_model(y, vol='Garch', p=1, q=1)

    # 모델 학습
    model_fit = model.fit(disp='off')

    # 한 스텝씩 예측
    y_pred = []
    for i in range(len(y)):
        forecast = model_fit.forecast(horizon=1, start=i)
        y_pred.append(forecast.mean.iloc[-1].values[0])

    return y, y_pred


# 성능 평가 함수
def evaluate_performance(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted, squared=False)
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

    # GARCH 모델 학습 및 예측
    actual, predicted = train_garch_model(data)

    # 성능 평가
    evaluate_performance(actual, predicted)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], actual, label='Actual Traffic(Q)', color='b')
    plt.plot(data['date'], predicted, label='Predicted Traffic(Q)', color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Traffic(Q)')
    plt.title('GARCH: Actual vs Predicted Traffic(Q)')
    plt.legend()
    plt.grid()
    plt.show()

    # df = pd.DataFrame({'value': predicted, 'real': actual})
    # df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/clock-compare/garch.csv', index=False)
else:
    print("지정된 디렉토리에 CSV 파일이 없습니다.")