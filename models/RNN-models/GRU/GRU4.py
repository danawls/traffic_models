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

# 교통 밀도, 속도, 흐름 계산
# data['density'] = data['통행속도']  # 예시로 속도를 밀도로 사용
# data['flow'] = data['density'] * data['통행속도']  # q_t = k_t * v_t

# # 교통량 변화율 계산
# data['delta_flow'] = data['통행속도'].diff()
#
# # 표준편차 계산
# sigma = data['delta_flow'].std()
#
# # 충격파 강도 계산
# data['shock_intensity'] = (data['delta_flow'].abs() > sigma).astype(int)
#
# # '통행속도' 정규화
# scaler = StandardScaler()
# data[['통행속도']] = scaler.fit_transform(
#     data[['통행속도']])


# 시퀀스 생성 함수
def create_sequences(data, seq_length=10):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length][['통행속도']].values
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


# 바닐라 GRU 모델 정의
class VanillaGRUModel(tf.keras.Model):
    def __init__(self, hidden_size, dropout_rate=0.2):
        super(VanillaGRUModel, self).__init__()
        self.gru = tf.keras.layers.GRU(hidden_size, dropout=dropout_rate, recurrent_dropout=dropout_rate,
                                       return_sequences=False)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.batch_norm(x)
        output = self.dense2(x)
        return output


# 모델 초기화
hidden_size = 64
vanilla_gru_model = VanillaGRUModel(hidden_size)

# 손실 함수와 옵티마이저 정의
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# 학습 함수 정의
@tf.function
def train_step_vanilla(sequences, targets):
    with tf.GradientTape() as tape:
        # 모델 실행
        predictions = vanilla_gru_model(sequences)

        # 손실 계산
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, vanilla_gru_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vanilla_gru_model.trainable_variables))
    return loss


# 학습 루프
EPOCHS = 100
for epoch in range(EPOCHS):
    total_loss = 0
    for sequences, targets in train_dataset:
        loss = train_step_vanilla(sequences, targets)
        total_loss += loss
    print(f"Epoch {epoch + 1}, Loss: {total_loss.numpy() / len(train_dataset):.4f}")


# 모델 평가 함수
def evaluate_model_vanilla(model, test_dataset):
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
predictions_vanilla, actuals_vanilla = evaluate_model_vanilla(vanilla_gru_model, test_dataset)

# 예측 시각화
plt.figure(figsize=(12, 6))
plt.plot(actuals_vanilla, label='Actual', color='b')
plt.plot(predictions_vanilla, label='Predicted', color='r')
plt.title('Actual vs Predicted (Vanilla GRU)')
plt.xlabel('Sample')
plt.ylabel('Normalized Speed')
plt.legend()
plt.show()