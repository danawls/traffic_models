import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
file_path = 'mnt/data/1.csv'
data = pd.read_csv(file_path)

# 'date' 컬럼을 datetime 형식으로 변환
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.sort_index()

# 교통 밀도, 속도, 흐름 계산
# data['density'] = data['통행속도']  # 예시로 속도를 밀도로 사용
# data['flow'] = data['density'] * data['통행속도']  # q_t = k_t * v_t

# 교통량 변화율 계산
data['delta_flow'] = data['통행속도'].diff()

# 표준편차 계산
sigma = data['delta_flow'].std()

# 충격파 강도 계산
data['shock_intensity'] = (data['delta_flow'].abs() > sigma).astype(int)

# '통행속도' 정규화
scaler = StandardScaler()
data[['통행속도', 'shock_intensity']] = scaler.fit_transform(
    data[['통행속도', 'shock_intensity']])


# 시퀀스 생성 함수
def create_sequences(data, seq_length=10):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length][['통행속도', 'shock_intensity']].values
        target = data.iloc[i + seq_length]['통행속도']
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


# 충격파 강도를 반영한 커스텀 GRU 모델 정의
class ShockwaveGRUModel(tf.keras.Model):
    def __init__(self, hidden_size):
        super(ShockwaveGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = tf.keras.layers.GRU(hidden_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(1)
        self.W_shock = tf.Variable(tf.random.normal([hidden_size, hidden_size]), trainable=True)

    def call(self, inputs):
        x, shock_intensity = inputs[:, :, :-1], inputs[:, :, -1]
        gru_out, state = self.gru(x)

        # 충격파 강도를 GRU에 반영
        shock_intensity = tf.expand_dims(shock_intensity, axis=-1)  # [batch_size, sequence_length, 1]
        shock_intensity = tf.tile(shock_intensity,
                                  [1, 1, self.hidden_size])  # [batch_size, sequence_length, hidden_size]
        shock_effect = tf.matmul(shock_intensity, self.W_shock)  # [batch_size, sequence_length, hidden_size]
        state = state + tf.reduce_mean(shock_effect, axis=1)  # [batch_size, hidden_size]

        # 예측 값 계산
        output = self.dense(state)
        return output


# 모델 초기화
hidden_size = 64
gru_model = ShockwaveGRUModel(hidden_size)

# 손실 함수와 옵티마이저 정의
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()


# 학습 함수 정의
@tf.function
def train_step(sequences, targets):
    with tf.GradientTape() as tape:
        # 모델 실행
        predictions = gru_model(sequences)

        # 손실 계산
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, gru_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, gru_model.trainable_variables))
    return loss


# 학습 루프
EPOCHS = 200
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

    # 성능 지표 계산
    mse = tf.reduce_mean(tf.keras.losses.MSE(actuals, predictions)).numpy()
    mae = tf.reduce_mean(tf.keras.losses.MAE(actuals, predictions)).numpy()
    rmse = np.sqrt(mse)

    # R2 계산
    ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
    ss_tot = np.sum((np.array(actuals) - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"Test MSE Loss: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")

    return predictions, actuals


# 모델 평가 및 예측
predictions, actuals = evaluate_model(gru_model, test_dataset)

# 예측 시각화
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual', color='b')
plt.plot(predictions, label='Predicted', color='r')
plt.title('Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Normalized Speed')
plt.legend()
plt.show()