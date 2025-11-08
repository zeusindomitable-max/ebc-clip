import torch
import torch.nn as nn
from ebc_clip import energy_budget_clip

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def test_ebc_clipping():
    model = MLP()
    x = torch.randn(2, 10)
    target = torch.randn(2, 1)
    
    clip_fn = energy_budget_clip(model)
    
    for _ in range(5):
        pred = model(x)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        clip_fn()
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        model.zero_grad()
    
    # Check no NaN
    assert not torch.isnan(model.fc1.weight).any()
