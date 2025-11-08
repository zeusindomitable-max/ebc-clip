# Toy MLP to show stability
import torch
import torch.nn as nn
from ebc_clip import energy_budget_clip

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    def forward(self, x): return self.net(x)

model = MLP()
x = torch.randn(32, 10)
y = torch.randn(32, 1)
clip_fn = energy_budget_clip(model)

for i in range(100):
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    clip_fn()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    model.zero_grad()
    if i % 20 == 0:
        print(f"Step {i} | Loss: {loss.item():.4f}")
