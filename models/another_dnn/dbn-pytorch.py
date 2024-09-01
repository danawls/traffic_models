import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import glob


# 데이터 전처리 함수
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 데이터 전처리
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data = data.sort_index()

    # 이상치 제거 및 필요한 컬럼만 유지
    data = data[(data['traffic(Q)'] > 0) & (data['traffic(Q)'] < data['traffic(Q)'].quantile(0.99))]
    data = data[(data['speed(u)'] > 0) & (data['speed(u)'] < data['speed(u)'].quantile(0.99))]

    return data


# Restricted Boltzmann Machine (RBM) 클래스 정의
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_visible))
        self.c = nn.Parameter(torch.zeros(n_hidden))

    def sample_from_p(self, p):
        return torch.bernoulli(p)

    def v_to_h(self, v):
        p_h = torch.sigmoid(torch.matmul(v, self.W.t()) + self.c)
        return p_h, self.sample_from_p(p_h)

    def h_to_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W) + self.b)
        return p_v, self.sample_from_p(p_v)

    def forward(self, v):
        p_h, h = self.v_to_h(v)
        p_v, v = self.h_to_v(h)
        return v

    def free_energy(self, v):
        vbias_term = torch.matmul(v, self.b)
        hidden_term = torch.sum(torch.log(1 + torch.exp(torch.matmul(v, self.W.t()) + self.c)), dim=1)
        return -vbias_term - hidden_term


# Deep Belief Network (DBN) 클래스 정의
class DBN(nn.Module):
    def __init__(self, rbm_layers, input_dim, output_dim):
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList([RBM(input_dim, rbm_layers[0])])
        for i in range(1, len(rbm_layers)):
            self.rbms.append(RBM(rbm_layers[i - 1], rbm_layers[i]))
        self.regressor = nn.Sequential(
            nn.Linear(rbm_layers[-1], 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        for rbm in self.rbms:
            _, x = rbm.v_to_h(x)
        return self.regressor(x)

    def pretrain(self, X_train, epochs=10, batch_size=64):
        for rbm in self.rbms:
            optimizer = optim.Adam(rbm.parameters(), lr=0.01)
            for epoch in range(epochs):
                epoch_error = 0
                for i in range(0, len(X_train), batch_size):
                    batch = X_train[i:i + batch_size]
                    batch = torch.tensor(batch, dtype=torch.float32)
                    if batch.shape[0] != batch_size:
                        continue
                    _, h = rbm.v_to_h(batch)
                    v_reconstructed, _ = rbm.h_to_v(h)
                    loss = torch.mean((batch - v_reconstructed) ** 2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_error += loss.item()
                print(f'RBM Pretrain Epoch: {epoch + 1}, Loss: {epoch_error / len(X_train):.4f}')


# 모델 학습 및 예측 함수
def train_dbn_model(data):
    # 입력 데이터와 타겟 데이터 정의
    X = data[['traffic(Q)', 'speed(u)', 'confusion', 'lane_number']].values
    y = data['traffic(Q)'].shift(-1).ffill().values  # 다음 시간대의 교통량을 타겟으로 설정

    # 데이터 스케일링
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 학습 데이터와 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # DBN 모델 초기화 및 사전 학습
    dbn = DBN(rbm_layers=[4, 32], input_dim=X_train.shape[1],
              output_dim=1)  # 수정: 첫 번째 RBM의 n_visible 값을 X_train.shape[1]과 맞춤
    dbn.pretrain(X_train, epochs=10, batch_size=32)

    # DBN 모델 학습
    optimizer = optim.Adam(dbn.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for epoch in range(100):
        dbn.train()
        optimizer.zero_grad()
        output = dbn(torch.tensor(X_train, dtype=torch.float32))
        loss = loss_fn(output.squeeze(), torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

    # 예측 수행
    dbn.eval()
    y_pred_scaled = dbn(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    return y_test_original, y_pred


# 성능 평가 함수
def evaluate_performance(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")


# 모든 CSV 파일을 불러오기 위한 경로 설정
file_paths = glob.glob('/Volumes/Expansion/traffic-prediction/product-data/con/6000VDS02200.csv')

# 첫 번째 CSV 파일로 모델 학습 및 평가
first_file = file_paths[0]
data = preprocess_data(first_file)

# DBN 모델 학습 및 예측
y_test, y_pred = train_dbn_model(data)

# 성능 평가
evaluate_performance(y_test, y_pred)

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Traffic(Q)', color='b')
plt.plot(y_pred, label='Predicted Traffic(Q)', color='r', linestyle='--')
plt.xlabel('Sample')
plt.ylabel('Traffic(Q)')
plt.title('DBN Model with PyTorch: Actual vs Predicted Traffic(Q)')
plt.legend()
plt.grid()
plt.show()