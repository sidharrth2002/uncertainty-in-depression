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
            packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.gru(packed_input)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, _ = self.gru(x)
        # Use the last hidden state
        h_n = output[:, -1, :]  # (batch_size, hidden_size * 2)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        var = torch.exp(logvar)
        return mu, var    

class COLDModel(nn.Module):
    def __init__(self, config: Config):
        super(COLDModel, self).__init__()

        self.audio_extractor = AudioFeatureExtractor()
        self.text_extractor = TextFeatureExtractor(config.vocab_size, config.embedding_dim, config.hidden_size)

        self.audio_temporal = LatentGRU()
        self.text_temporal = LatentGRU()


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

        text_features = self.text_extractor(text_features)

        mean_text, variance_text = self.text_temporal(text_features)

        mean_modalities = [mean_audio, mean_text]
        variance_modalities = [variance_audio, variance_text]

        weights = self.compute_fusion_weights(variance_modalities)

        fused_mean = sum(weight * mean for weight, mean in zip(weights, mean_modalities))

        y_pred = self.output_layer(fused_mean)

        return y_pred, mean_modalities, variance_modalities