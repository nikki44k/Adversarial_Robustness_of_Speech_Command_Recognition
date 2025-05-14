# models.py  (replace your existing class with this)

import torch.nn as nn

class Conv1DSpeech(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # 0
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            # 1
            nn.BatchNorm1d(16),
            # 2
            nn.ReLU(),

            # 3
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            # 4
            nn.BatchNorm1d(32),
            # 5
            nn.ReLU(),

            # 6
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            # 7
            nn.BatchNorm1d(64),
            # 8
            nn.ReLU(),

            # 9
            nn.AdaptiveAvgPool1d(1),
            #10
            nn.Flatten(),
            #11  ← this placeholder pushes your Linear to idx 12
            nn.Identity(),
            #12
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)
