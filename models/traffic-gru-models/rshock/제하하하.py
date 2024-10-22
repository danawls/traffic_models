import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.models import Model

PRED_RATIO = 0.2


# CSV 파일을 읽어옵니다.
file_path = f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/10/6000VDS03500.csv'
data = pd.read_csv(file_path)

# 필요한 열 선택
traffic_data = data['traffic(Q)'].values
speed_data = data['speed(u)'].values

# 충격파 강도 계산 (연속적인 traffic(Q) 값의 변화율로 근사)
delta_traffic = np.diff(traffic_data, prepend=traffic_data[0])
delta_speed = np.diff(speed_data, prepend=speed_data[0])
shock_wave_intensity = np.divide(delta_traffic, delta_speed, out=np.zeros_like(delta_traffic, dtype=float), where=delta_speed != 0)

# 새로운 열 'shock_wave_intensity' 추가
data['shock_wave_intensity'] = shock_wave_intensity

# 입력 데이터와 타겟 데이터 설정
features = ['speed(u)', 'confusion', 'lane_number', 'shock_wave_intensity']
target = 'traffic(Q)'

# 입력 데이터와 타겟 데이터 정규화
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data[features])

# 시계열 데이터를 위한 입력 및 출력 데이터 생성 함수
def create_sequences(data, target, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = target[i + seq_length]  # traffic(Q)의 다음 값을 예측
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 시퀀스 길이 설정
sequence_length = 10

# 타겟 데이터 정규화 (출력만 정규화된 데이터로 예측해야 함)
target_data = scaler.fit_transform(data[[target]])

# 시퀀스 데이터 생성
X, y = create_sequences(normalized_data, target_data, sequence_length)

# 학습, 검증, 테스트 데이터셋 분할 (80%, 10%, 10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# TensorFlow에서 사용자 정의 GRU 층을 구현하기 위해 사용자 정의 층을 만듭니다.
class ShockwaveGRU(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ShockwaveGRU, self).__init__(**kwargs)
        self.units = units
        self.gru = tf.keras.layers.GRU(units, return_state=True)

    def build(self, input_shape):
        # W_shock은 스칼라 형태의 가중치로 초기화
        self.W_shock = self.add_weight(shape=(1, 1),  # 스칼라 값으로 변경
                                       initializer='random_normal',
                                       trainable=True, name='W_shock')
        super(ShockwaveGRU, self).build(input_shape)

    def call(self, inputs, states):
        shock_intensity = inputs[..., -1:]  # 마지막 입력이 shock wave intensity라고 가정
        # 스칼라 곱셈으로 shock_contribution 계산
        shock_contribution = shock_intensity * self.W_shock
        # 기존 입력과 shock_contribution을 결합하여 modulated_input을 만듭니다.
        modulated_input = tf.concat([inputs[..., :-1], shock_contribution], axis=-1)
        output, state = self.gru(modulated_input, initial_state=states)
        return output, state

# 모델 입력 정의
input_layer = Input(shape=(sequence_length, X_train.shape[2]))

# 사용자 정의 ShockwaveGRU 층 사용
gru_layer, state = ShockwaveGRU(64)(input_layer, states=None)

# 출력층
output_layer = Dense(1)(gru_layer)

# 모델 생성
model = Model(inputs=input_layer, outputs=output_layer)

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 모델 학습
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val))

# 모델 평가
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss (MSE): {test_loss}')





file_path = f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/1/6000VDS03500.csv'
data = pd.read_csv(file_path)

data = data.iloc[:int((len(data) * PRED_RATIO)) - 1, :]

# 필요한 열 선택
traffic_data = data['traffic(Q)'].values
speed_data = data['speed(u)'].values

# 충격파 강도 계산 (연속적인 traffic(Q) 값의 변화율로 근사)
delta_traffic = np.diff(traffic_data, prepend=traffic_data[0])
delta_speed = np.diff(speed_data, prepend=speed_data[0])
shock_wave_intensity = np.divide(delta_traffic, delta_speed, out=np.zeros_like(delta_traffic, dtype=float), where=delta_speed != 0)

# 새로운 열 'shock_wave_intensity' 추가
data['shock_wave_intensity'] = shock_wave_intensity

# 입력 데이터와 타겟 데이터 설정
features = ['speed(u)', 'confusion', 'lane_number', 'shock_wave_intensity']
target = 'traffic(Q)'
normalized_data = scaler.fit_transform(data[features])
target_data = scaler.fit_transform(data[[target]])
X, y = create_sequences(normalized_data, target_data, sequence_length)
# 예측
y_pred = model.predict(X)

y_pred_original = scaler.inverse_transform(y_pred)
y_test_original = scaler.inverse_transform(y)

# 성능 지표 계산
mae = mean_absolute_error(y_test_original, y_pred_original )
mse = mean_squared_error(y_test_original, y_pred_original )
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original ))
r2 = r2_score(y_test_original, y_pred_original )
mape = mean_absolute_percentage_error(y_test_original, y_pred_original )

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')

# 결과 출력
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='Actual', color='b')
plt.plot(y_pred_original, label='Predicted', color='r')
plt.title('Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Speed')
plt.legend()
plt.show()

# df = pd.DataFrame({'value': list(y_pred_original), 'real':list(y_test_original)})
# df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/deep-compare/rshock5.csv', index=False)