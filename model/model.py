import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, hp):
        super(CRNN, self).__init__()
        self.hp = hp

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(3,3)), 
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(3,3)),

            nn.Conv2d(32, 128, kernel_size=(3,3), stride=(1,1), padding=(3,3)),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(3,3))
        )

        self.lstm = nn.LSTM(
            hp.model.lstm_input,
            hp.model.lstm_dim,
            batch_first=True)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.fc = nn.Linear(hp.model.lstm_output, hp.model.fc_dim)
 
    def forward(self, x):
        # x: [B, T, num_freq]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_freq]
        x = self.conv(x)
        # x: [B, 8, T, num_freq]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]

        x, _ = self.lstm(x)  # [B, T, lstm_dim]
        x = self.maxpool(x)
        x = F.relu(x)
        x = torch.reshape(x, (x.size(0), x.size(1)*x.size(2)))
        x = self.fc(x)  # x: [B, T, fc_dim]
        return x
