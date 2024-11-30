from dataclasses import dataclass
import torch
import torch.nn as nn
from datatypes import *
from feature_extractors import *

class LatentGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LatentGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_size * 2, output_size)
        self.fc_logvar = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len, input_size)
        if lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.gru(packed_input)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            output, _ = self.gru(x)
        # Use the last hidden state
        h_n = output[:, -1, :]  # (batch_size, hidden_size * 2)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        var = torch.exp(logvar)
        return mu, var


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

if __name__ == "__main__":
    model = COLDModel()
    print(model)
    # count number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")