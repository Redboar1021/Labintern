import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# データの生成 (サンプル)
np.random.seed(0)
data = np.sin(np.linspace(0, 20, 200)) + np.random.normal(scale=0.1, size=200)  # 正常データ
data[70:85] += 2  # 異常データの挿入

# データをPyTorchのテンソルに変換
data = torch.tensor(data, dtype=torch.float32)

# モデルの定義
class DDPM(nn.Module):
    def __init__(self):
        super(DDPM, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

model = DDPM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# モデルのトレーニング
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(data[:-1].unsqueeze(-1))
    loss = criterion(outputs, data[1:].unsqueeze(-1))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 異常検出
model.eval()
with torch.no_grad():
    predictions = model(data.unsqueeze(-1)).squeeze()
    errors = torch.abs(predictions - data)

# 異常部分の表示
threshold = errors.mean() + 2 * errors.std()  # 閾値を設定
anomalies = data[errors > threshold]

plt.figure(figsize=(10, 6))
plt.plot(data.numpy(), label='Data')
plt.plot(predictions.numpy(), label='Reconstructed')
plt.scatter(anomalies.numpy().nonzero(), anomalies.numpy(), color='red', label='Anomalies')
plt.legend()
plt.show()
