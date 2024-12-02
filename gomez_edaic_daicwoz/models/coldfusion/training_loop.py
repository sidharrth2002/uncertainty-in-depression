from loss_functions import *
from model import COLDModelSimplified
import torch
import torch.optim as optim

# Initialize Model, Optimizer
output_size = 2  # e.g., 0: Not Depressed, 1: Depressed
model = COLDModelSimplified(output_size=output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Hyperparameters for loss weighting (can be tuned)
lambda_COV = 1e-3
lambda_COA = 1e-3
lambda_COAV = 1e-3
lambda_R = 1e-4

# Training Loop
num_epochs = 10  # Example number of epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:  # Replace with your DataLoader
        # Unpack batch
        audio_features, visual_features, labels = batch  # Ensure correct unpacking
        audio_features = audio_features.to(device)
        visual_features = visual_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward Pass
        y_pred_fused, y_pred_audio, y_pred_visual, mean_modalities, variance_modalities = model(audio_features, visual_features)
        # y_pred_fused: (batch_size, output_size)
        # y_pred_audio: (batch_size, output_size)
        # y_pred_visual: (batch_size, output_size)
        # mean_modalities: list of [mean_audio, mean_visual]
        # variance_modalities: list of [var_audio, var_visual]

        # Emotion Prediction Loss (Classification)
        L_emo = emotion_prediction_loss_classification(y_pred_fused, labels)

        # Compute Distance Vectors (Per-Modality Cross-Entropy Loss)
        criterion = nn.CrossEntropyLoss(reduction='none')
        dA = criterion(y_pred_audio, labels).unsqueeze(1)  # Shape: (batch_size, 1)
        dV = criterion(y_pred_visual, labels).unsqueeze(1)  # Shape: (batch_size, 1)
        # If Text modality is included, compute dT similarly

        # Compute Variance-Norm Vectors (Inverse L2 Norms)
        sA = 1.0 / (torch.norm(variance_modalities[0], p=2, dim=1, keepdim=True) + 1e-8)  # Shape: (batch_size, 1)
        sV = 1.0 / (torch.norm(variance_modalities[1], p=2, dim=1, keepdim=True) + 1e-8)  # Shape: (batch_size, 1)
        # If Text modality is included, compute sT similarly

        # Create Lists for Calibration and Ordinality Loss
        distance_vectors_list = [dA, dV]  # Add dT if Text is included
        variance_norm_vectors_list = [sA, sV]  # Add sT if Text is included

        # Compute Calibration and Ordinality Loss
        L_CO = softmax_distributional_matching_loss(distance_vectors_list, variance_norm_vectors_list)

        # Compute Variance Regularization Loss for Each Modality
        L_regu_A = variance_regularization_loss(mean_modalities[0], variance_modalities[0])  # Audio
        L_regu_V = variance_regularization_loss(mean_modalities[1], variance_modalities[1])  # Visual
        # If Text modality is included, compute L_regu_T similarly
        L_regu = L_regu_A + L_regu_V  # + L_regu_T if applicable

        # Organize Calibration and Ordinality Loss Components
        # Distribute equally if no crossmodal constraints are defined
        L_CO_dict = {
            'COV': L_CO,    # Visual-only Calibration and Ordinality loss
            'COA': L_CO,    # Audio-only Calibration and Ordinality loss
            'COAV': L_CO    # Crossmodal Calibration and Ordinality loss (if applicable)
        }

        # Compute Total Loss
        L_total = total_loss_function(L_emo, L_CO_dict, L_regu, 
                                      lambda_COV=lambda_COV, 
                                      lambda_COA=lambda_COA, 
                                      lambda_COAV=lambda_COAV, 
                                      lambda_R=lambda_R)

        # Backward Pass and Optimization
        L_total.backward()
        optimizer.step()

        running_loss += L_total.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
