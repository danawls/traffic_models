import tensorflow as tf
from tensorflow.keras.layers import Layer, GRUCell
import numpy as np


class MultiPhaseGRUCell(GRUCell):
    def __init__(self, units, state_matrix_size, **kwargs):
        super(MultiPhaseGRUCell, self).__init__(units, **kwargs)
        self.state_matrix_size = state_matrix_size
        # 상태 영향(P) 및 상태 전이 영향(H)의 가중치
        self.P = self.add_weight(shape=(state_matrix_size,), initializer="random_normal", trainable=True,
                                 name="P_matrix")
        self.H = self.add_weight(shape=(state_matrix_size, state_matrix_size), initializer="random_normal",
                                 trainable=True, name="H_matrix")

    def call(self, inputs, states):
        prev_hidden_state = states[0]

        # 상태 전이 확률 행렬(S) 소프트맥스 출력을 통한 확률 도출
        S = tf.nn.softmax(inputs)  # 가정: inputs은 상태 전이 확률

        # 상태 결정
        current_state = tf.argmax(S, axis=-1)
        P_current = tf.gather(self.P, current_state)

        # 상태 전이 영향 반영
        prev_state = tf.argmax(tf.nn.softmax(prev_hidden_state), axis=-1)
        H_transition = tf.gather(self.H[prev_state], current_state)

        # 기본 GRU 업데이트
        output, new_states = super(MultiPhaseGRUCell, self).call(inputs, states)

        # P_current와 H_transition을 은닉 상태에 반영
        updated_hidden_state = output + P_current + H_transition

        return updated_hidden_state, [updated_hidden_state]


# 상태 크기 지정 (임의로 설정)
state_matrix_size = 5


def build_multi_phase_gru_model(input_shape, units, state_matrix_size):
    inputs = tf.keras.Input(shape=input_shape)

    # 커스텀 GRUCell 사용
    gru_layer = tf.keras.layers.RNN(MultiPhaseGRUCell(units, state_matrix_size), return_sequences=True)(inputs)

    # 출력 레이어
    outputs = tf.keras.layers.Dense(1, activation='linear')(gru_layer)

    model = tf.keras.Model(inputs, outputs)

    # 컴파일
    model.compile(optimizer='adam', loss='mse')

    return model


# 모델 빌드
input_shape = (10, 3)  # 예시 입력 크기
units = 64
model = build_multi_phase_gru_model(input_shape, units, state_matrix_size)

model.summary()

# 학습 데이터 준비
X_train = np.random.rand(100, 10, 3)  # 예시 데이터
y_train = np.random.rand(100, 10, 1)

# 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=32)