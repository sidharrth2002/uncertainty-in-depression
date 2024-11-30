"""
Code from the original repository:
https://github.com/PingCheng-Wei/DepressionEstimation/blob/main/models/AVT_ConvLSTM/models/convlstm.py
"""

import math
import torch
import torch.nn as nn


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, height) = layer.weight.size()
        n = n_in * height
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_lstm(layer):
    """
    Initialises the hidden layers in the LSTM - H0 and C0.

    Input
        layer: torch.Tensor - The LSTM layer
    """
    n_i1, n_i2 = layer.weight_ih_l0.size()
    n_i = n_i1 * n_i2

    std = math.sqrt(2. / n_i)
    scale = std * math.sqrt(3.)
    layer.weight_ih_l0.data.uniform_(-scale, scale)

    if layer.bias_ih_l0 is not None:
        layer.bias_ih_l0.data.fill_(0.)

    n_h1, n_h2 = layer.weight_hh_l0.size()
    n_h = n_h1 * n_h2

    std = math.sqrt(2. / n_h)
    scale = std * math.sqrt(3.)
    layer.weight_hh_l0.data.uniform_(-scale, scale)

    if layer.bias_hh_l0 is not None:
        layer.bias_hh_l0.data.fill_(0.)


def init_att_layer(layer):
    """
    Initilise the weights and bias of the attention layer to 1 and 0
    respectively. This is because the first iteration through the attention
    mechanism should weight each time step equally.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    layer.weight.data.fill_(1.)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    Initialize a Batchnorm layer.

    Input
        bn: torch.Tensor - The batch normalisation layer
    """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvLSTM_Visual(nn.Module):
    def __init__(self, input_dim, output_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout):
        super(ConvLSTM_Visual, self).__init__()
        self.conv = ConvBlock2d(in_channels=input_dim,
                                out_channels=conv_hidden,
                                kernel=(72, 3),
                                stride=(1, 1),
                                pad=(0, 1),
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=conv_hidden,
                            hidden_size=lstm_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.fc = FullyConnected(in_channels=lstm_hidden*2,
                                 out_channels=output_dim,
                                 activation=activation,
                                 normalisation=norm)

    def forward(self, net_input):
        x = net_input
        batch, C, F, T = x.shape
        x = self.conv(x)
        x = self.pool(x.squeeze())
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)                                 # output shape: (batch, width//stride(pool), lstm_hidden*2) 5x600x128
        x = self.fc(x[:, -1, :].reshape(batch, -1))         # output shape: (batch, output_dim)

        return x


class ConvLSTM_Audio(nn.Module):
    def __init__(self, input_dim, output_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout):
        super(ConvLSTM_Audio, self).__init__()
        self.conv = ConvBlock1d(in_channels=input_dim,      # 80
                                out_channels=conv_hidden,   # 128
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')         # ['bn', 'wn', else]
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(dropout)                     # 0.2
        self.lstm = nn.LSTM(input_size=conv_hidden,         # 128
                            hidden_size=lstm_hidden,        # 128
                            num_layers=num_layers,          # 2
                            batch_first=True,
                            bidirectional=True)
        self.fc = FullyConnected(in_channels=lstm_hidden*2,   # 128
                                 out_channels=output_dim,   # 2
                                 activation=activation,     # ['sigmoid', 'softmax', 'global', else]
                                 normalisation=norm)        # ['bn', 'wn']: nn.BatchNorm1d, nn.utils.weight_norm

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)                                 # output shape: (batch, width//stride(pool), lstm_hidden*2) 5x600x128
        x = self.fc(x[:, -1, :].reshape(batch, -1))         # output shape: (batch, output_dim)

        return x


class ConvLSTM_Text(nn.Module):
    def __init__(self, input_dim, output_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout):
        super(ConvLSTM_Text, self).__init__()
        self.conv = ConvBlock1d(in_channels=input_dim,
                                out_channels=conv_hidden,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')         # ['bn', 'wn', else]
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=conv_hidden,
                            hidden_size=lstm_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.fc = FullyConnected(in_channels=lstm_hidden*2,
                                 out_channels=output_dim,
                                 activation=activation,     # ['sigmoid', 'softmax', 'global', else]
                                 normalisation=norm)        # ['bn', 'wn']: nn.BatchNorm1d, nn.utils.weight_norm

    def forward(self, net_input):
        x = net_input
        batch, F, T = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)                                 # output shape: (batch, width//stride(pool), lstm_hidden*2) 5x600x128
        x = self.fc(x[:, -1, :].reshape(batch, -1))         # output shape: (batch, output_dim)

        return x

# class AudioFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(AudioFeatureExtractor, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.relu(self.conv3(x))
#         x = self.pool(x)
#         x = self.relu(self.conv4(x))
#         x = self.pool(x)
#         return x
    
# class TextFeatureExtractor(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_size):
#         super(TextFeatureExtractor, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)

#     def forward(self, x):
#         x = self.embedding(x)
#         x, _ = self.lstm(x)
#         return x

class COLDModel(nn.Module):
    def __init__(self):
        super(COLDModel, self).__init__()

        config = {
            "visual_net": {
                "input_dim": 3,
                "output_dim": 256,
                "conv_hidden": 256,
                "lstm_hidden": 256,
                "num_layers": 4,
                "activation": "relu",
                "norm": "bn",
                "dropout": 0.6,
            },
            "audio_net": {
                "input_dim": 80,
                "output_dim": 256,
                "conv_hidden": 256,
                "lstm_hidden": 256,
                "num_layers": 4,
                "activation": "relu",
                "norm": "bn",
                "dropout": 0.6,
            },
            "text_net": {
                "input_dim": 512,
                "output_dim": 256,
                "conv_hidden": 256,
                "lstm_hidden": 256,
                "num_layers": 2,
                "activation": "relu",
                "norm": "bn",
                "dropout": 0.6,
            },
        }

        self.audio_extractor = ConvLSTM_Audio(
            input_dim=config["audio_net"]["input_dim"],
            output_dim=config["audio_net"]["output_dim"],
            conv_hidden=config["audio_net"]["conv_hidden"],
            lstm_hidden=config["audio_net"]["lstm_hidden"],
            num_layers=config["audio_net"]["num_layers"],
            activation=config["audio_net"]["activation"],
            norm=config["audio_net"]["norm"],
            dropout=config["audio_net"]["dropout"],
        )

        # self.text_extractor = ConvLSTM_Text(
        #     input_dim=config["text_net"]["input_dim"],
        #     output_dim=config["text_net"]["output_dim"],
        #     conv_hidden=config["text_net"]["conv_hidden"],
        #     lstm_hidden=config["text_net"]["lstm_hidden"],
        #     num_layers=config["text_net"]["num_layers"],
        #     activation=config["text_net"]["activation"],
        #     norm=config["text_net"]["norm"],
        #     dropout=config["text_net"]["dropout"],
        # )

        self.visual_extractor = ConvLSTM_Visual(
            input_dim=config["visual_net"]["input_dim"],
            output_dim=config["visual_net"]["output_dim"],
            conv_hidden=config["visual_net"]["conv_hidden"],
            lstm_hidden=config["visual_net"]["lstm_hidden"],
            num_layers=config["visual_net"]["num_layers"],
            activation=config["visual_net"]["activation"],
            norm=config["visual_net"]["norm"],
            dropout=config["visual_net"]["dropout"],
        )

        # GRU used to estimate the mean and variance of the latent space
        self.audio_temporal = LatentGRU(
            input_size=config["audio_net"]["output_dim"],
            hidden_size=256,
            output_size=128
        )
        # self.text_temporal = LatentGRU(
        #     input_size=config["text_net"]["output_dim"],
        #     hidden_size=256,
        #     output_size=128
        # )
        self.visual_temporal = LatentGRU(
            input_size=config["visual_net"]["output_dim"],
            hidden_size=256,
            output_size=128
        )

    
    def compute_fusion_weights(self, variance_modalities):
        """
        Compute fusion weights based on the variance of each modality.
        """
        norms = [v.norm(p=2, dim=1) for v in variance_modalities]

        # reciprocal to give more weight to modalities with lower variance
        inv_norms = [1 / norm for norm in norms]
        total_inv_norms = sum(inv_norms)
        weights = [inv_norm / total_inv_norms for inv_norm in inv_norms]

        weights = [weight.unsqueeze(1) for weight in weights]

        return weights

    def forward(self, audio_features, text_features):
        audio_features = self.audio_extractor(audio_features)
        mean_audio, variance_audio = self.audio_temporal(audio_features)

        # text_features = self.text_extractor(text_features)
        # mean_text, variance_text = self.text_temporal(text_features)

        visual_features = self.visual_extractor(visual_features)
        mean_visual, variance_visual = self.visual_temporal(visual_features)

        mean_modalities = [mean_audio, mean_visual]
        variance_modalities = [variance_audio, variance_visual]

        weights = self.compute_fusion_weights(variance_modalities)

        fused_mean = sum(
            weight * mean for weight, mean in zip(weights, mean_modalities)
        )

        y_pred = self.output_layer(fused_mean)

        return y_pred, mean_modalities, variance_modalities

