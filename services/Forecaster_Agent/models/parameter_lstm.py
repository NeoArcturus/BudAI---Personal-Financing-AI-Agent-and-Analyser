import torch

import torch.nn as nn

from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

class ParameterLSTM(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=9):

        super(ParameterLSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        _, (hn, _) = self.lstm(x)

        out = self.fc(hn[-1])

        return torch.sigmoid(out)
