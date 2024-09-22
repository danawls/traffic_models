import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기 (사용자 데이터 경로로 변경)
data = pd.read_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/1/6000VDS02200.csv')

# 'traffic(Q)' 열을 시계열 데이터로 사용
traffic_data = data['traffic(Q)'].values

# 교통 상태를 분류하기 위한 임계값 정의 (임의로 설정)
# 이 임계값은 데이터의 특성에 맞게 조정해야 합니다.
low_traffic = 50
medium_traffic = 100

# 교통 상태를 범주형으로 변환
def categorize_traffic(traffic):
    if traffic < low_traffic:
        return 'Low'
    elif traffic < medium_traffic:
        return 'Medium'
    else:
        return 'High'

traffic_states = np.array([categorize_traffic(t) for t in traffic_data])

# 상태 인코딩
label_encoder = LabelEncoder()
state_encoded = label_encoder.fit_transform(traffic_states)

# 전이 행렬 초기화
n_states = len(label_encoder.classes_)
transition_matrix = np.zeros((n_states, n_states))

# 전이 행렬 계산
for (i, j) in zip(state_encoded[:-1], state_encoded[1:]):
    transition_matrix[i][j] += 1

# 각 상태에서 전이 확률로 정규화
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

print("Transition Matrix:")
print(transition_matrix)

# 마르코프 체인 예측 함수
def predict_next_state(current_state, transition_matrix):
    return np.random.choice(n_states, p=transition_matrix[current_state])

# 현재 상태를 기반으로 다음 10개의 상태 예측
current_state = state_encoded[0]  # 첫 번째 상태에서 시작
predicted_states = [current_state]

for _ in range(10):
    next_state = predict_next_state(current_state, transition_matrix)
    predicted_states.append(next_state)
    current_state = next_state

# 예측된 상태를 원래 상태 이름으로 디코딩
predicted_state_names = label_encoder.inverse_transform(predicted_states)

print("Predicted States:")
print(predicted_state_names)

# 실제 상태와 예측 상태 시각화
plt.figure(figsize=(12, 6))
plt.plot(range(len(traffic_states)), traffic_states, label='Actual Traffic State')
plt.plot(range(len(predicted_state_names)), predicted_state_names, label='Predicted Traffic State', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Traffic State')
plt.title('Traffic State Prediction using Markov Chain')
plt.legend()
plt.show()