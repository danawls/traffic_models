import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.vector_ar.var_model import VAR
import glob

t = 1

# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 날짜를 인덱스로 설정하고 정렬
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    # 필요한 컬럼 선택
    data = data[['date', 'traffic(Q)', 'speed(u)', 'confusion']]

    # 인덱스를 날짜로 설정
    data.set_index('date', inplace=True)

    # 결측값이 있는 경우 처리 (여기서는 선형 보간 사용)
    data = data.interpolate(method='linear')

    return data


# VAR 모델 학습 및 예측 함수
def train_var_model(data, forecast_steps=5):
    # VAR 모델은 다변량 시계열 데이터를 사용하므로 다수의 시계열을 선택해야 합니다
    model = VAR(data)

    # 최적의 lag 선택
    lag_order = model.select_order()
    model_fitted = model.fit(lag_order.aic)

    # 예측
    forecast = model_fitted.forecast(data.values[-lag_order.aic:], steps=forecast_steps)

    # 예측 결과를 데이터프레임으로 변환
    forecast_df = pd.DataFrame(forecast,
                               index=pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='H')[1:],
                               columns=data.columns)

    return forecast_df


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
    print(f"Mean Absolute Percentae Error (MAPE): {mape:.4f}")

# 모든 CSV 파일을 불러오기 위한 경로 설정
file_paths = glob.glob(f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/{t}/6000VDS03500.csv')

# 첫 번째 CSV 파일로 모델 학습 및 평가
if file_paths:  # 파일이 존재하는지 확인
    first_file = file_paths[0]
    data = preprocess_data(first_file)

    # VAR 모델 학습 및 예측
    forecast_steps = len(data)
    forecast_df = train_var_model(data, forecast_steps=forecast_steps)

    # 실제 값과 예측 값 비교 (여기서는 'traffic(Q)' 컬럼에 대해서만)
    actual = data['traffic(Q)'][-forecast_steps:].values
    predicted = forecast_df['traffic(Q)'].values

    # 성능 평가
    evaluate_performance(actual, predicted)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Traffic(Q)', color='b')
    plt.plot(predicted, label='Predicted Traffic(Q)', color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Traffic(Q)')
    plt.title('VAR: Actual vs Predicted Traffic(Q)')
    plt.legend()
    plt.grid()
    plt.show()

    # df = pd.DataFrame({'value': predicted, 'real': actual})
    # df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/clock-compare/var.csv', index=False)
else:
    print("지정된 디렉토리에 CSV 파일이 없습니다.")