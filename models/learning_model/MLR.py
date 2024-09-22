# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
# 'data.csv' 파일을 사용한다고 가정합니다. 각자의 데이터에 맞게 경로와 파일명을 수정하세요.
data = pd.read_csv(f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/1/6000VDS03500.csv')

# 데이터 확인
print(data.head())

# 독립 변수(X)와 종속 변수(y)를 정의합니다.
# 종속 변수는 다중 클래스 값이어야 합니다. 예시에서는 교통 상태를 범주형으로 설정.
X = data[['speed(u)', 'confusion', 'lane_number']]  # 독립 변수
y = data['traffic(Q)'].astype('category')  # 종속 변수: 교통 상태 (범주형으로 변환)

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 다중 로지스틱 회귀 모델 생성 및 학습 (multinomial 옵션 사용)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 분류 보고서 출력
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 혼동 행렬 출력 및 시각화
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 모델 회귀 계수 출력
print(f"회귀 계수 (Coefficients):\n{model.coef_}")
print(f"절편 (Intercept): {model.intercept_}")