# 노드의 특성으로는 교통량, 속도, 혼잡빈도 등을 사용할 수 있습니다.
# 엣지는 시간 흐름에 따른 연결을 나타내며, 노드 간의 인접성을 정의합니다.
# GNN 모델은 이러한 노드와 엣지 정보를 학습하여 다음 시간대의 교통량을 예측합니다.

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import matplotlib.pyplot as plt


# 데이터 전처리 함수 정의
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')

    # 필요한 피처 선택 및 정규화
    features = ['traffic(Q)', 'speed(u)', 'confusion', 'lane_number']
    x_f = ['speed(u)', 'confusion', 'lane_number']
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    data[x_f] = scaler_x.fit_transform(data[x_f])
    data['traffic(Q)'] = scaler_y.fit_transform(data[['traffic(Q)']])

    return data, scaler_x, scaler_y


# 그래프 데이터 생성 함수
def create_graph_data(data):
    x = torch.tensor(data[['speed(u)', 'confusion', 'lane_number']].values, dtype=torch.float)
    edge_index = torch.tensor([[i, i + 1] for i in range(len(data) - 1)], dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 양방향 그래프
    y = torch.tensor(data['traffic(Q)'].shift(-1).fillna(0).values, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


# GNN 모델 정의
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# 모델 학습 함수
def train_gnn_model(data_list, epochs=500, batch_size=64):
    model = GNNModel(input_dim=3, hidden_dim=16, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = F.mse_loss(output[:-1].view(-1), batch.y[:-1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')

    return model


# 모델 평가 함수
def evaluate_model(model, data_list):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in data_list:
            output = model(data)
            y_true.append(data.y[:-1].numpy())
            y_pred.append(output[:-1].view(-1).numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f'MSE: {mse}, MAE: {mae}, R2: {r2}, mse:{rmse}, mape:{mape}')

    return y_true, y_pred


# 시각화 함수
def plot_results(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual Traffic(Q)')
    plt.plot(y_pred, label='Predicted Traffic(Q)', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Traffic(Q)')
    plt.title('Actual vs Predicted Traffic(Q)')
    plt.legend()
    plt.show()


# 데이터 로드 및 전처리
data_files = [f'/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/test_data/10/6000VDS03500.csv']
data_list = []

for file in data_files:
    data, scaler_x, scaler_y = preprocess_data(file)
    graph_data = create_graph_data(data)
    data_list.append(graph_data)

# 모델 학습
model = train_gnn_model(data_list)

# 모델 평가 및 결과 시각화
y_true, y_pred = evaluate_model(model, data_list)
plot_results(y_true, y_pred)

# df = pd.DataFrame({'value': list(y_pred), 'real': y_true})
# df.to_csv('/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/table/graph-compare/gnn.csv', index=False)