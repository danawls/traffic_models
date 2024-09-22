
import tensorflow as tf
import numpy as np

# 커스텀 GRU 셀 정의
class MultiPhaseGRUCell(tf.keras.layers.GRUCell):
    def __init__(self, units, state_matrix_size, **kwargs):
        super(MultiPhaseGRUCell, self).__init__(units, **kwargs)
        self.state_matrix_size = state_matrix_size
        # 상태 영향(P) 및 상태 전이 영향(H)의 가중치
        self.P = self.add_weight(shape=(state_matrix_size,), initializer="random_normal", trainable=True, name="P_matrix")
        self.H = self.add_weight(shape=(state_matrix_size, state_matrix_size), initializer="random_normal", trainable=True, name="H_matrix")

    def call(self, inputs, states):
        prev_hidden_state = states[0]
        # 상태 전이 확률 행렬(S) 소프트맥스 출력을 통한 확률 도출
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

# 예시 입력 데이터 크기 및 유닛 설정
input_shape = (10, 3)
units = 64

# 모델 빌드 및 요약
model = build_multi_phase_gru_model(input_shape, units, state_matrix_size)
model.summary()

# 학습 데이터 준비 (예시 데이터)
X_train = np.random.rand(100, 10, 3)
y_train = np.random.rand(100, 10, 1)

# 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=32)
