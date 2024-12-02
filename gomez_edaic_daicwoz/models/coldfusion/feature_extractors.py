"""
Instead of using an additional GRU, we directly use the LSTM to output the mean and variance.
"""

import math
import torch
import torch.nn as nn
from common_layers import *


class ConvLSTM_Visual(nn.Module):
    def __init__(self, input_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout, output_size):
        super(ConvLSTM_Visual, self).__init__()
        self.conv = ConvBlock2d(in_channels=input_dim,
                                out_channels=conv_hidden,
                                kernel=(3, 3),
                                stride=(1, 1),
                                pad=(1, 1),
                                normalisation=norm)
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=conv_hidden,
                            hidden_size=lstm_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        # Output layers for mean and variance
        self.fc_mu = nn.Linear(lstm_hidden * 2, output_size)
        self.fc_logvar = nn.Linear(lstm_hidden * 2, output_size)
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the LSTM and FC layers."""
        init_layer(self.fc_mu)
        init_layer(self.fc_logvar)

    def forward(self, net_input):
        """
        Forward pass through the ConvLSTM_Visual module.
        Args:
            net_input: (batch_size, channels, freq, time)
        Returns:
            mu: (batch_size, output_size)
            var: (batch_size, output_size)
        """
        x = net_input
        # batch, C, F, T = x.shape
        x = self.conv(x)
        batch, conv_hidden, F_prime, T_prime = x.shape
        x = x.view(batch, conv_hidden, F_prime * T_prime)  # Correct
        x = self.pool(x)              # (batch, conv_hidden, new_time)
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()  # (batch, new_time, conv_hidden)
        x, _ = self.lstm(x)                  # (batch, new_time, lstm_hidden*2)
        x = x[:, -1, :]                      # (batch, lstm_hidden*2)
        mu = self.fc_mu(x)                   # (batch, output_size)
        logvar = self.fc_logvar(x)           # (batch, output_size)
        var = torch.exp(logvar)              # (batch, output_size)
        return mu, var


class ConvLSTM_Audio(nn.Module):
    def __init__(self, input_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout, output_size):
        super(ConvLSTM_Audio, self).__init__()
        self.conv = ConvBlock1d(in_channels=input_dim,      # e.g., 80
                                out_channels=conv_hidden,   # e.g., 128
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation=norm)
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(dropout)                     # e.g., 0.2
        self.lstm = nn.LSTM(input_size=conv_hidden,         # e.g., 128
                            hidden_size=lstm_hidden,        # e.g., 128
                            num_layers=num_layers,          # e.g., 2
                            batch_first=True,
                            bidirectional=True)
        # Output layers for mean and variance
        self.fc_mu = nn.Linear(lstm_hidden * 2, output_size)
        self.fc_logvar = nn.Linear(lstm_hidden * 2, output_size)
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the LSTM and FC layers."""
        init_layer(self.fc_mu)
        init_layer(self.fc_logvar)

    def forward(self, net_input):
        """
        Forward pass through the ConvLSTM_Audio module.
        Args:
            net_input: (batch_size, freq, time)
        Returns:
            mu: (batch_size, output_size)
            var: (batch_size, output_size)
        """
        x = net_input
        batch, F, T = x.shape
        x = self.conv(x)
        x = self.pool(x)          # (batch, conv_hidden, new_time)
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()  # (batch, new_time, conv_hidden)
        x, _ = self.lstm(x)                  # (batch, new_time, lstm_hidden*2)
        x = x[:, -1, :]                      # (batch, lstm_hidden*2)
        mu = self.fc_mu(x)                   # (batch, output_size)
        logvar = self.fc_logvar(x)           # (batch, output_size)
        var = torch.exp(logvar)              # (batch, output_size)
        return mu, var


class ConvLSTM_Text(nn.Module):
    def __init__(self, input_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout, output_size):
        super(ConvLSTM_Text, self).__init__()
        self.conv = ConvBlock1d(in_channels=input_dim,
                                out_channels=conv_hidden,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation=norm)         # ['bn', 'wn', else]
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=conv_hidden,
                            hidden_size=lstm_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        # Output layers for mean and variance
        self.fc_mu = nn.Linear(lstm_hidden * 2, output_size)
        self.fc_logvar = nn.Linear(lstm_hidden * 2, output_size)
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the LSTM and FC layers."""
        init_layer(self.fc_mu)
        init_layer(self.fc_logvar)

    def forward(self, net_input):
        """
        Forward pass through the ConvLSTM_Text module.
        Args:
            net_input: (batch_size, F, T)
        Returns:
            mu: (batch_size, output_size)
            var: (batch_size, output_size)
        """
        x = net_input
        batch, F, T = x.shape
        x = self.conv(x)
        x = self.pool(x)          # (batch, conv_hidden, new_time)
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()  # (batch, new_time, conv_hidden)
        x, _ = self.lstm(x)                  # (batch, new_time, lstm_hidden*2)
        x = x[:, -1, :]                      # (batch, lstm_hidden*2)
        mu = self.fc_mu(x)                   # (batch, output_size)
        logvar = self.fc_logvar(x)           # (batch, output_size)
        var = torch.exp(logvar)              # (batch, output_size)
        return mu, var
