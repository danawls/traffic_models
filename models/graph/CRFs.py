import pandas as pd
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split

# 데이터 불러오기
data = pd.read_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/con/6000VDS02200.csv')

# 시퀀스를 일정한 길이로 나누기 위한 함수
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(0, len(data) - seq_length, seq_length):
        X.append(extract_features(data.iloc[i:i + seq_length]))
        y.append(extract_labels(data.iloc[i:i + seq_length]))
    return X, y

# 특성 추출 함수
def extract_features(data):
    features = []
    for i in range(len(data)):
        features.append({
            'traffic(Q)': data['traffic(Q)'].iloc[i],
            'speed(u)': data['speed(u)'].iloc[i],
            'confusion': data['confusion'].iloc[i],
            'lane_number': data['lane_number'].iloc[i]
        })
    return features

# 라벨 추출 함수
def extract_labels(data):
    return data['traffic(Q)'].shift(-1).ffill().astype(int).astype(str).tolist()  # 문자열로 변환

# 시퀀스 길이 설정
sequence_length = 5  # 적절한 시퀀스 길이로 설정하세요.

# 시퀀스 데이터 생성
X_seq, y_seq = create_sequences(data, sequence_length)

# 학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# CRF 모델 정의
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

# 모델 학습
crf.fit(X_train, y_train)

# 예측
y_pred = crf.predict(X_test)

# 성능 평가
print(flat_classification_report(y_test, y_pred))