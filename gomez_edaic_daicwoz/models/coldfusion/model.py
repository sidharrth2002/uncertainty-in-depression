from dataclasses import dataclass
import time
import torch
import torch.nn as nn
from datatypes import *
# from original_feature_extractors import *
from feature_extractors import *
import torch.nn.functional as F

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

class COLDModelSimplified(nn.Module):
    def __init__(self, output_size=2):
        super(COLDModelSimplified, self).__init__()

        config = {
            "visual_net": {
                "input_dim": 3,
                "conv_hidden": 256,
                "lstm_hidden": 256,
                "num_layers": 4,
                "activation": "relu",
                "norm": "bn",
                "dropout": 0.6,
                "output_size": 128,
            },
            "audio_net": {
                "input_dim": 80,
                "conv_hidden": 256,
                "lstm_hidden": 256,
                "num_layers": 4,
                "activation": "relu",
                "norm": "bn",
                "dropout": 0.6,
                "output_size": 128,
            },
            "text_net": {
                "input_dim": 512,
                "conv_hidden": 256,
                "lstm_hidden": 256,
                "num_layers": 2,
                "activation": "relu",
                "norm": "bn",
                "dropout": 0.6,
                "output_size": 128,
            },
        }

        # TODO: if using pre-trained model, load the frozen weights here
        self.audio_extractor = ConvLSTM_Audio(
            input_dim=config["audio_net"]["input_dim"],
            conv_hidden=config["audio_net"]["conv_hidden"],
            lstm_hidden=config["audio_net"]["lstm_hidden"],
            num_layers=config["audio_net"]["num_layers"],
            activation=config["audio_net"]["activation"],
            norm=config["audio_net"]["norm"],
            dropout=config["audio_net"]["dropout"],
            output_size=config["audio_net"]["output_size"],
        )

        self.visual_extractor = ConvLSTM_Visual(
            input_dim=config["visual_net"]["input_dim"],
            conv_hidden=config["visual_net"]["conv_hidden"],
            lstm_hidden=config["visual_net"]["lstm_hidden"],
            num_layers=config["visual_net"]["num_layers"],
            activation=config["visual_net"]["activation"],
            norm=config["visual_net"]["norm"],
            dropout=config["visual_net"]["dropout"],
            output_size=config["visual_net"]["output_size"],
        )

        # If text modality is used, initialize similarly
        # self.text_extractor = ConvLSTM_Text(
        #     input_dim=config["text_net"]["input_dim"],
        #     conv_hidden=config["text_net"]["conv_hidden"],
        #     lstm_hidden=config["text_net"]["lstm_hidden"],
        #     num_layers=config["text_net"]["num_layers"],
        #     activation=config["text_net"]["activation"],
        #     norm=config["text_net"]["norm"],
        #     dropout=config["text_net"]["dropout"],
        #     output_size=config["text_net"]["output_size"],
        # )

        # Output layer
        self.output_layer = FullyConnected(
            in_channels=128,  # Number of modalities
            out_channels=output_size,
            activation='softmax',  # For classification
            normalisation=None
        )

    def compute_fusion_weights(self, variance_modalities):
        """
        Compute fusion weights based on the variance of each modality.
        Lower variance -> higher weight
        """
        # Compute L2 norms of variances
        norms = [v.norm(p=2, dim=1) for v in variance_modalities]  # List of (batch, )

        # Convert to weights: inverse of norm
        inv_norms = [1.0 / (norm + 1e-8) for norm in norms]  # Avoid division by zero

        # Stack and apply softmax to get weights
        stacked_inv_norms = torch.stack(inv_norms, dim=1)  # (batch, num_modalities)
        weights = F.softmax(stacked_inv_norms, dim=1)     # (batch, num_modalities)

        return weights  # (batch, num_modalities)

    def forward(self, batch, modalities = ["edaic_audio_mfcc", "edaic_video_pose_gaze_aus"]):
        """
        Forward pass through the COLDModelSimplified.
        Args:
            audio_features: (batch_size, freq, time)
            visual_features: (batch_size, channels, freq, time)
        Returns:
            y_pred: (batch_size, output_size)
            mean_modalities: list of (batch_size, output_size)
            variance_modalities: list of (batch_size, output_size)
        """
        
        
            # if "edaic_video_cnn_resnet" in modality_names:
            #     return ["edaic_video_cnn_resnet"]

            # if "edaic_video_pose_gaze_aus" in modality_names:
            #     return ["edaic_video_pose_gaze_aus"]

            # if "edaic_audio_mfcc" in modality_names:
            #     return ["edaic_audio_mfcc"]

            # if "edaic_audio_egemaps" in modality_names:
            #     return ["edaic_audio_egemaps"]

            # if "edaic_audio_densenet201" in modality_names:
            #     return ["edaic_audio_densenet201"]

            # if "edaic_audio_vgg16" in modality_names:
            #     return ["edaic_audio_vgg16"]
        
        all_audio_data = []
        all_visual_data = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        
        for modality in modalities:
            modality_id = modality
            
            data = batch[f"modality:{modality_id}:data"]
            mask = batch[f"modality:{modality_id}:mask"]
            
            if modality_id == "edaic_audio_mfcc":
                all_audio_data.append(data)
            elif modality_id == "edaic_video_pose_gaze_aus":
                all_visual_data.append(data)
        
        # Extract features and uncertainty
        
        start_time = time.time()
        
        all_audio_data = torch.tensor(all_audio_data[0]).to(device)
        all_visual_data = torch.tensor(all_visual_data[0]).to(device)
        
        mean_audio, var_audio = self.audio_extractor(all_audio_data)        # Each: (batch, output_size)
        mean_visual, var_visual = self.visual_extractor(all_visual_data)   # Each: (batch, output_size)
        
        # mean_audio, var_audio = self.audio_extractor(data["modality:edaic_audio_mfcc:data"])        # Each: (batch, output_size)
        # mean_visual, var_visual = self.visual_extractor(data["modality:edaic_video_pose_gaze_aus:data"])   # Each: (batch, output_size)

        # If text modality is used, process similarly
        # mean_text, var_text = self.text_extractor(text_features)

        mean_modalities = [mean_audio, mean_visual]  # Extend if text is included
        variance_modalities = [var_audio, var_visual]

        # Compute fusion weights based on variance
        weights = self.compute_fusion_weights(variance_modalities)  # (batch, num_modalities)

        # Stack means
        stacked_means = torch.stack(mean_modalities, dim=1)  # (batch, num_modalities, output_size)

        # Apply weights
        weights = weights.unsqueeze(2)  # (batch, num_modalities, 1)
        weighted_means = stacked_means * weights  # (batch, num_modalities, output_size)

        # Sum over modalities to get fused representation
        fused_mean = weighted_means.sum(dim=1)  # (batch, output_size)

        # Pass through output layer
        y_pred = self.output_layer(fused_mean)  # (batch, output_size)

        return y_pred, mean_modalities, variance_modalities


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = COLDModelSimplified(output_size=2)  # e.g., 2 classes for depression detection
    model.to(device)
    print(model)
    # Count number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    # simulate a batch of data
    batch = {
        "modality:edaic_audio_mfcc:data": torch.randn(4, 80, 300),
        "modality:edaic_audio_mfcc:mask": torch.randn(4, 1),
        "modality:edaic_video_pose_gaze_aus:data": torch.randn(4, 3, 72, 3),
        "modality:edaic_video_pose_gaze_aus:mask": torch.randn(4, 1),
    }
    
    y_pred, mean_modalities, variance_modalities = model(batch)
    print(f"y_pred shape: {y_pred.shape}")