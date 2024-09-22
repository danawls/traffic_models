
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 데이터 로드 및 전처리
file_path = '/path/to/your/data.csv'  # 실제 파일 경로로 교체
data = pd.read_csv(file_path)

# 'date' 컬럼을 datetime 형식으로 변환
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.sort_index()

# 현재 상태 변수 S
data['state'] = pd.cut(data['speed(u)'], bins=[-np.inf, 45, 70, np.inf], labels=[0, 1, 2])

# 시퀀스 생성 함수
def create_sequences(data, seq_length=10):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length][['speed(u)', 'confusion', 'lane_number', 'state']].values
        target = data.iloc[i + seq_length]['traffic(Q)']
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# 시퀀스 및 타겟 생성
sequences, targets = create_sequences(data.dropna())

# 학습 및 테스트 데이터 분할
train_sequences, test_sequences, train_targets, test_targets = train_test_split(sequences, targets, test_size=0.2, random_state=42)

# 커스텀 GRU 셀 정의
class MultiPhaseGRUCell(tf.keras.layers.GRUCell):
    def __init__(self, units, state_matrix_size, **kwargs):
        super(MultiPhaseGRUCell, self).__init__(units, **kwargs)
        self.state_matrix_size = state_matrix_size
        self.P = self.add_weight(shape=(state_matrix_size,), initializer="random_normal", trainable=True, name="P_matrix")
        self.H = self.add_weight(shape=(state_matrix_size, state_matrix_size), initializer="random_normal", trainable=True, name="H_matrix")

    def call(self, inputs, states):
        prev_hidden_state = states[0]
        S = tf.nn.softmax(inputs)
        current_state = tf.argmax(S, axis=-1)
        P_current = tf.gather(self.P, current_state)
        prev_state = tf.argmax(tf.nn.softmax(prev_hidden_state), axis=-1)
        H_transition = tf.gather(self.H[prev_state], current_state)
        output, new_states = super(MultiPhaseGRUCell, self).call(inputs, states)
        updated_hidden_state = output + P_current + H_transition
        return updated_hidden_state, [updated_hidden_state]

# 상태 크기 지정 (임의로 설정)
state_matrix_size = 5

# 모델 빌드 함수
def build_multi_phase_gru_model(input_shape, units, state_matrix_size):
    inputs = tf.keras.Input(shape=input_shape)
    gru_layer = tf.keras.layers.RNN(MultiPhaseGRUCell(units, state_matrix_size), return_sequences=True)(inputs)
    outputs = tf.keras.layers.Dense(1, activation='linear')(gru_layer)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# 모델 빌드
input_shape = (train_sequences.shape[1], train_sequences.shape[2])
units = 64
model = build_multi_phase_gru_model(input_shape, units, state_matrix_size)

# 모델 요약 출력
model.summary()

# TensorFlow Dataset 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_targets)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_targets)).batch(32)

# 모델 학습
model.fit(train_dataset, epochs=10, batch_size=32, validation_data=test_dataset)

# 평가 및 예측
predictions = model.predict(test_dataset)
mse = mean_squared_error(test_targets, predictions)
mae = mean_absolute_error(test_targets, predictions)
r2 = r2_score(test_targets, predictions)

print(f'MSE: {mse}, MAE: {mae}, R2 Score: {r2}')
