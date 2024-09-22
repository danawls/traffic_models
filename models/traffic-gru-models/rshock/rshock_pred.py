import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import glob
import pickle

input_ratio = 0.5

file_path = pd.read_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/1/6000VDE02602.csv')

pickle_path = 'rshock_p.pkl'

data = pd.read_csv(file_path)

length = data.shape[0] * input_ratio

data = data.iloc[:(length - 1), :]

# 'date' 컬럼을 datetime 형식으로 변환
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.sort_index()

# 교통 밀도, 속도, 흐름 계산
# data['density'] = data['통행속도']  # 예시로 속도를 밀도로 사용
# data['flow'] = data['density'] * data['통행속도']  # q_t = k_t * v_t

# 교통량 변화율 계산
data['delta_flow'] = data['traffic(Q)'].diff()

# 표준편차 계산
sigma = data['delta_flow'].std()

# 충격파 강도 계산
data['shock_intensity'] = (data['delta_flow'].abs() > sigma).astype(int)

# 기존 모델 불러오기 및 파인튜닝
with open(pickle_path, 'rb') as f:
    model = pickle.load(f)

model.predict(data)

