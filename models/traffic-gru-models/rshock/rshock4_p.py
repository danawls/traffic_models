import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import glob
import pickle

HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCH = 500
PRED_RATIO = 1

# 시퀀스 생성 함수
def create_sequences(data, seq_length=10):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length][['speed(u)', 'confusion', 'lane_number', 'shock_intensity']].values
        target = data.iloc[i + seq_length]['traffic(Q)']
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


class MovingAvgGRUModel(tf.keras.Model):
    def __init__(self, hidden_size, dropout_rate=DROPOUT):
        super(MovingAvgGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = tf.keras.layers.GRU(hidden_size, return_sequences=True, return_state=True,
                                       dropout=dropout_rate,
                                       recurrent_dropout=dropout_rate)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(1)
        self.W_shock = tf.Variable(tf.random.normal([1]), trainable=True)

    def call(self, inputs):
        x, shock_intensity = inputs[:, :, :-1], inputs[:, :, -1]
        gru_out, state = self.gru(x)

        # 충격파 강도를 GRU에 반영
        shock_intensity = tf.expand_dims(shock_intensity, axis=-1)  # [batch_size, sequence_length, 1]
        shock_intensity = tf.tile(shock_intensity,
                                  [1, 1, self.hidden_size])  # [batch_size, sequence_length, hidden_size]
        shock_effect = shock_intensity * self.W_shock  # [batch_size, sequence_length, hidden_size]
        state = state + tf.reduce_mean(shock_effect, axis=1)  # [batch_size, hidden_size]

        # Dense 레이어 및 드롭아웃, 배치 정규화 추가
        x = self.dense1(state)
        x = self.dropout(x)
        x = self.batch_norm(x)
        output = self.dense2(x)
        return output

# 모델 초기화
hidden_size = 64  # 은닉 크기 증가
moving_avg_gru_model = MovingAvgGRUModel(hidden_size)

# 손실 함수와 옵티마이저 정의
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


# 학습 함수 정의
@tf.function
def train_step(sequences, targets):
    with tf.GradientTape() as tape:
        # 모델 실행
        predictions = moving_avg_gru_model(sequences)

        # 손실 계산
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, moving_avg_gru_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, moving_avg_gru_model.trainable_variables))
    return loss


for k in range(1):
    tm = 1 + 3 * k

    file_lists = sorted(
        glob.glob(f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/1/*.csv'))
    pickle_path = 'rshock4_p.pkl'

    # file_list = file_lists[:len(file_lists) - 2]
    # pred_file = file_lists[len(file_lists) - 1]

    file_list = file_lists[-3:]
    pred_file = f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/1/6000VDS03500.csv'
    for i, file_path in enumerate(file_list):
        data = pd.read_csv(file_path)

        # 'date' 컬럼을 datetime 형식으로 변환
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        data = data.sort_index()

        # 교통 밀도, 속도, 흐름 계산
        # data['density'] = data['통행속도']  # 예시로 속도를 밀도로 사용
        # data['flow'] = data['density'] * data['통행속도']  # q_t = k_t * v_t

        # 이동 평균 계산
        window_size = 18  # 이동 평균을 계산할 윈도우 크기
        data['moving_avg_flow'] = data['speed(u)'].rolling(window=window_size).mean()

        # 교통량 변화율 계산
        data['delta_flow'] = data['speed(u)'] - data['moving_avg_flow']

        # 표준편차를 이용한 임계값 설정
        k = 0.7  # 임계값 조정을 위한 상수
        sigma = k * data['delta_flow'].std()

        # 충격파 강도 계산
        data['shock_intensity'] = (data['delta_flow'].abs() > sigma).astype(int)

        # # '통행속도' 정규화
        # scaler = StandardScaler()
        # data[['통행속도', 'shock_intensity']] = scaler.fit_transform(
        #     data[['통행속도', 'shock_intensity']])

        # 시퀀스 및 타겟 생성
        sequences, targets = create_sequences(data.dropna())  # 결측값 제거

        # 학습 및 테스트 데이터 분할
        train_sequences, test_sequences, train_targets, test_targets = train_test_split(sequences, targets,
                                                                                        test_size=0.2,
                                                                                        random_state=42)

        # TensorFlow Dataset 생성
        train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_targets)).batch(BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_targets)).batch(BATCH_SIZE)

        # 학습 루프
        EPOCHS = EPOCH  # 학습 에폭
        for epoch in range(EPOCHS):
            total_loss = 0
            for sequences, targets in train_dataset:
                loss = train_step(sequences, targets)
                total_loss += loss
            print(f"Epoch {epoch + 1}, Loss: {total_loss.numpy() / len(train_dataset):.4f}")


        # 모델 평가 함수
        def evaluate_model(model, test_dataset):
            predictions = []
            actuals = []
            for sequences, targets in test_dataset:
                preds = model(sequences, training=False)
                predictions.extend(preds.numpy().flatten())
                actuals.extend(targets.numpy())

            test_mse = mean_squared_error(actuals, predictions)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(actuals, predictions)
            test_r2 = r2_score(actuals, predictions)

            print(f"Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test R2: {test_r2}")

            return predictions, actuals


        # 모델 평가 및 예측
        predictions, actuals = evaluate_model(moving_avg_gru_model, test_dataset)

        # 예측 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label='Actual', color='b')
        plt.plot(predictions, label='Predicted', color='r')
        plt.title('Actual vs Predicted (Moving Average GRU)')
        plt.xlabel('Sample')
        plt.ylabel('Speed')
        plt.legend()
        plt.show()

        # 모델 저장 (업데이트)
        with open(pickle_path, 'wb') as f:
            pickle.dump((moving_avg_gru_model), f)



data = pd.read_csv(pred_file)

data = data.iloc[:int((len(data) * PRED_RATIO)) - 1, :]

# 'date' 컬럼을 datetime 형식으로 변환
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.sort_index()

# 교통 밀도, 속도, 흐름 계산
# data['density'] = data['통행속도']  # 예시로 속도를 밀도로 사용
# data['flow'] = data['density'] * data['통행속도']  # q_t = k_t * v_t

# 이동 평균 계산
window_size = 18  # 이동 평균을 계산할 윈도우 크기
data['moving_avg_flow'] = data['speed(u)'].rolling(window=window_size).mean()

# 교통량 변화율 계산
data['delta_flow'] = data['speed(u)'] - data['moving_avg_flow']

# 표준편차를 이용한 임계값 설정
k = 0.7  # 임계값 조정을 위한 상수
sigma = k * data['delta_flow'].std()

# 충격파 강도 계산
data['shock_intensity'] = (data['delta_flow'].abs() > sigma).astype(int)

# # '통행속도' 정규화
# scaler = StandardScaler()
# data[['통행속도', 'shock_intensity']] = scaler.fit_transform(
#     data[['통행속도', 'shock_intensity']])

# 시퀀스 및 타겟 생성
sequences, targets = create_sequences(data.dropna())  # 결측값 제거

predictions = []
actuals = []

preds = moving_avg_gru_model(sequences, training=False)
predictions = list(preds)
actuals = list(targets)

test_mse = mean_squared_error(actuals, predictions)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(actuals, predictions)
test_r2 = r2_score(actuals, predictions)
test_mape = mean_absolute_percentage_error(actuals, predictions)

print(
    f"Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test R2: {test_r2}, Test MAPE: {test_mape}")

# 예측 시각화
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual', color='b')
plt.plot(predictions, label='Predicted', color='r')
plt.title('Actual vs Predicted (Moving Average GRU)')
plt.xlabel('Sample')
plt.ylabel('Speed')
plt.legend()
plt.show()

df = pd.DataFrame({'value': predictions, 'real':actuals})
df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/deep-compare/rshock4.csv', index=False)