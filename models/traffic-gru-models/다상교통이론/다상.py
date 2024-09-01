import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 데이터 로드 및 전처리
file_path = '/Volumes/Expansion/traffic-prediction/product-data/1/32.csv'
data = pd.read_csv(file_path)

# 'date' 컬럼을 datetime 형식으로 변환
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.sort_index()

# '통행속도' 정규화
scaler = StandardScaler()
data['통행속도'] = scaler.fit_transform(data[['통행속도']])


# 시퀀스 생성 함수
def create_sequences(data, seq_length=10):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length]['통행속도'].values
        target = data.iloc[i + seq_length]['통행속도']
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


# 시퀀스 및 타겟 생성
sequences, targets = create_sequences(data)

# 학습 및 테스트 데이터 분할
train_sequences, test_sequences, train_targets, test_targets = train_test_split(sequences, targets, test_size=0.2,
                                                                                random_state=42)

# TensorFlow Dataset 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_targets)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_targets)).batch(32)


# GRU 모델 정의
class GRUStateModel(tf.keras.Model):
    def __init__(self, hidden_size, num_classes):
        super(GRUStateModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = tf.keras.layers.GRU(hidden_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(num_classes)
        self.P = tf.Variable(tf.random.normal([num_classes, hidden_size]), trainable=True)
        self.H = tf.Variable(tf.random.normal([num_classes, num_classes, hidden_size]), trainable=True)

    def call(self, inputs, training=False):
        x, prev_state, prev_state_index = inputs
        gru_out, state = self.gru(x, initial_state=prev_state)
        state_output = self.dense(gru_out[:, -1, :])

        # 상태 전이 확률 계산
        S = tf.nn.softmax(state_output, axis=1)

        # 현재 상태 결정
        current_state_index = tf.argmax(S, axis=1, output_type=tf.int32)  # int32로 명시적 변환

        # 상태 영향 행렬 P 적용
        P_current = tf.gather(self.P, current_state_index)

        # 상태 전이 영향 행렬 H 적용
        if prev_state_index is not None:
            H_transition = tf.gather_nd(self.H, tf.stack([prev_state_index, current_state_index], axis=1))
        else:
            H_transition = tf.zeros_like(P_current)

        # 은닉 상태 수정
        new_state = state + P_current + H_transition

        return new_state, S, current_state_index


# 모델 초기화
hidden_size = 64
num_classes = 5  # 상태 수
model = GRUStateModel(hidden_size, num_classes)

# 손실 함수와 옵티마이저 정의
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()


# 학습 함수 정의
@tf.function
def train_step(sequences, targets):
    with tf.GradientTape() as tape:
        # 초기 상태 설정
        initial_state = tf.zeros((sequences.shape[0], hidden_size))
        prev_state_index = tf.zeros((sequences.shape[0],), dtype=tf.int32)

        # 모델 실행
        state, S, current_state_index = model((tf.expand_dims(sequences, -1), initial_state, prev_state_index))

        # 손실 계산
        loss = loss_object(targets, tf.squeeze(state[:, -1]))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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
        initial_state = tf.zeros((sequences.shape[0], hidden_size))
        prev_state_index = tf.zeros((sequences.shape[0],), dtype=tf.int32)
        state, S, current_state_index = model((tf.expand_dims(sequences, -1), initial_state, prev_state_index),
                                              training=False)
        predictions.extend(tf.squeeze(state[:, -1]).numpy())
        actuals.extend(targets.numpy())
    test_mse = mean_squared_error(actuals, predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(actuals, predictions)
    test_r2 = r2_score(actuals, predictions)

    # Output performance metrics
    print(f"Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test R2: {test_r2}")
    return predictions, actuals


# 모델 평가 및 예측
predictions, actuals = evaluate_model(model, test_dataset)

# 예측 시각화
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual', color='b')
plt.plot(predictions, label='Predicted', color='r')
plt.title('Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Normalized Speed')
plt.legend()
plt.show()