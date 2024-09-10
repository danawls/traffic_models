import numpy as np
import pandas as pd
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

# 데이터 불러오기
data = pd.read_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/con/6000VDS02200.csv')

# 필요한 열 선택
traffic_flow = data['traffic(Q)'].values  # 교통량
speed = data['speed(u)'].values  # 속도

# 간단한 이산화 처리 (예: 교통량이 임계값보다 높은지 낮은지를 기준으로)
traffic_flow_discrete = np.digitize(traffic_flow, bins=[0, 100, 200, 300, 400])
speed_discrete = np.digitize(speed, bins=[0, 20, 40, 60, 80])

# MRF 모델 정의
mrf_model = MarkovModel()

# 노드 추가 (교통량과 속도를 노드로 추가)
mrf_model.add_nodes_from(['TrafficFlow', 'Speed'])

# 엣지 추가 (교통량과 속도 간의 관계를 정의)
mrf_model.add_edges_from([('TrafficFlow', 'Speed')])

# 잠재 변수 (Potential) 정의 - 이 예제에서는 단순히 임의의 잠재 변수를 사용합니다.
traffic_flow_potential = DiscreteFactor(['TrafficFlow'], [5], np.random.rand(5))
speed_potential = DiscreteFactor(['Speed'], [5], np.random.rand(5))
joint_potential = DiscreteFactor(['TrafficFlow', 'Speed'], [5, 5], np.random.rand(25))

# 잠재 변수를 모델에 추가
mrf_model.add_factors(traffic_flow_potential, speed_potential, joint_potential)

# 신뢰도 전파(Belief Propagation) 사용하여 추론
belief_propagation = BeliefPropagation(mrf_model)

# 증거 설정 및 추론
evidence = {'Speed': 2}  # 속도가 특정 상태인 경우
result = belief_propagation.map_query(variables=['TrafficFlow'], evidence=evidence)

print("추론 결과 (교통량 상태):", result)