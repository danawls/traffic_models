import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# 데이터 로드 및 전처리
file_path = f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/1/6000VDS03500.csv'
data = pd.read_csv(file_path)

# 'date' 컬럼을 datetime 형식으로 변환
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.sort_index()

# 교통 밀도, 속도, 흐름 계산
# data['density'] = data['통행속도']  # 예시로 속도를 밀도로 사용
# data['flow'] = data['density'] * data['통행속도']  # q_t = k_t * v_t

# 입력 데이터 변화량 계산
data['delta_x'] = data['speed(u)'].diff()

# 뉴웰의 관성 모델을 위한 반응 시간 설정
K = 3  # 3 스텝 뒤에 반응한다고 가정

# 입력 데이터 변화량을 K타임 스텝 뒤로 이동하여 반응 시간 고려
data['delta_x_shifted'] = data['delta_x'].shift(K)


# 시퀀스 생성 함수
def create_sequences(data, seq_length=10):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length][['speed(u)', 'confusion', 'lane_number', 'delta_x_shifted']].values
        target = data.iloc[i + seq_length]['traffic(Q)']
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


# 시퀀스 및 타겟 생성
sequences, targets = create_sequences(data.dropna())  # 결측값 제거

# 학습 및 테스트 데이터 분할
train_sequences, test_sequences, train_targets, test_targets = train_test_split(sequences, targets, test_size=0.2,
                                                                                random_state=42)

# TensorFlow Dataset 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_targets)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_targets)).batch(32)


# 뉴웰의 관성 모델을 반영한 GRU 모델 정의
class NewellGRUModel(tf.keras.Model):
    def __init__(self, hidden_size, dropout_rate=0.2):
        super(NewellGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = tf.keras.layers.GRU(hidden_size, return_sequences=True, return_state=True, dropout=dropout_rate,
                                       recurrent_dropout=dropout_rate)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(1)
        self.T = tf.Variable(tf.random.normal([1]), trainable=True)  # 변화량의 가중치

    def call(self, inputs):
        x, delta_x_shifted = inputs[:, :, :-1], inputs[:, :, -1]
        gru_out, state = self.gru(x)

        # 리셋 게이트와 업데이트 게이트에 변화량 반영
        delta_x_effect = self.T * tf.expand_dims(delta_x_shifted, axis=-1)  # [batch_size, sequence_length, 1]
        delta_x_effect = tf.tile(delta_x_effect, [1, 1, self.hidden_size])  # [batch_size, sequence_length, hidden_size]

        # 은닉 상태에 변화량의 효과 반영
        state = state + tf.reduce_mean(delta_x_effect, axis=1)  # [batch_size, hidden_size]

        # Dense 레이어 및 드롭아웃, 배치 정규화 추가
        x = self.dense1(state)
        x = self.dropout(x)
        x = self.batch_norm(x)
        output = self.dense2(x)
        return output


# 모델 초기화
hidden_size = 64  # 은닉 크기 증가
newell_gru_model = NewellGRUModel(hidden_size)

# 손실 함수와 옵티마이저 정의
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# 학습 함수 정의
@tf.function
def train_step(sequences, targets):
    with tf.GradientTape() as tape:
        # 모델 실행
        predictions = newell_gru_model(sequences)

        # 손실 계산
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, newell_gru_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, newell_gru_model.trainable_variables))
    return loss


# 학습 루프
EPOCHS = 500  # 학습 에폭
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
predictions, actuals = evaluate_model(newell_gru_model, test_dataset)

# 예측 시각화
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual', color='b')
plt.plot(predictions, label='Predicted', color='r')
plt.title('Actual vs Predicted (Newell GRU)')
plt.xlabel('Sample')
plt.ylabel('Normalized Speed')
plt.legend()
plt.show()




file_path = '/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/1/6000VDE02302.csv'
data = pd.read_csv(file_path)

# 'date' 컬럼을 datetime 형식으로 변환
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.sort_index()

# 교통 밀도, 속도, 흐름 계산
# data['density'] = data['통행속도']  # 예시로 속도를 밀도로 사용
# data['flow'] = data['density'] * data['통행속도']  # q_t = k_t * v_t

# 입력 데이터 변화량 계산
data['delta_x'] = data['speed(u)'].diff()

# 뉴웰의 관성 모델을 위한 반응 시간 설정
K = 3  # 3 스텝 뒤에 반응한다고 가정

# 입력 데이터 변화량을 K타임 스텝 뒤로 이동하여 반응 시간 고려
data['delta_x_shifted'] = data['delta_x'].shift(K)

sequences, targets = create_sequences(data.dropna())

predictions = []
actuals = []

preds = newell_gru_model(sequences, training=False)
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
plt.title('Actual vs Predicted (Newell GRU)')
plt.xlabel('Sample')
plt.ylabel('Speed')
plt.legend()
plt.show()