import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob


# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 날짜를 인덱스로 설정하고 정렬
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    # 교통량을 분류하기 위한 임의의 다중 클래스 생성
    bins = [0, 50, 100, 150, np.inf]
    labels = [0, 1, 2, 3]  # 다중 클래스 레이블
    data['traffic_class'] = pd.cut(data['traffic(Q)'], bins=bins, labels=labels)

    # 데이터 타입을 적절히 변환하여 결측값 처리 가능하도록 함
    for column in ['traffic(Q)', 'speed(u)', 'confusion']:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # 결측값이 있는 경우 처리 (여기서는 선형 보간 사용)
    data[['traffic(Q)', 'speed(u)', 'confusion']] = data[['traffic(Q)', 'speed(u)', 'confusion']].interpolate(
        method='linear')

    # 입력(X)과 출력(y) 정의
    X = data[['speed(u)', 'confusion']].values
    y = data['traffic_class'].astype(int).values  # 다중 클래스 레이블

    return X, y


# 모델 학습 및 예측 함수
def train_mnr_model(X, y):
    """ 주어진 데이터에 대해 MNR 모델을 학습하고 예측합니다. """
    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 훈련 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 다항 로지스틱 회귀 모델 정의 및 학습
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    return y_test, y_pred, model


# 모델 평가 함수
def evaluate_performance(y_test, y_pred):
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# 모든 CSV 파일을 불러오기 위한 경로 설정
file_paths = glob.glob('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/1/6000VDS02200.csv')

# 첫 번째 CSV 파일로 모델 학습 및 평가
if file_paths:  # 파일이 존재하는지 확인
    first_file = file_paths[0]
    X, y = preprocess_data(first_file)

    # Multinomial Logistic Regression 모델 학습 및 예측
    y_test, y_pred, model = train_mnr_model(X, y)

    # 성능 평가 및 시각화
    evaluate_performance(y_test, y_pred)

else:
    print("지정된 디렉토리에 CSV 파일이 없습니다.")