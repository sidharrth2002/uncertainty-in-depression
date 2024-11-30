import torch
import torch.nn as nn
import torch.nn.functional as F

# Emotion Prediction Loss for Classification
def emotion_prediction_loss_classification(y_pred, labels):
    """
    Computes the Cross-Entropy Loss for classification tasks.

    Args:
        y_pred (torch.Tensor): Logits from the model. Shape: (batch_size, num_classes)
        labels (torch.Tensor): Ground truth labels. Shape: (batch_size,)

    Returns:
        torch.Tensor: Scalar loss value.
    """
    criterion = nn.CrossEntropyLoss()
    return criterion(y_pred, labels)

# Calibration and Ordinality Loss
def calibration_ordinal_loss(distance_vectors, variance_norm_vectors):
    """
    Computes the Calibration and Ordinality Loss using KL Divergence
    between softmax distributions of distance vectors and variance-norm vectors.

    Args:
        distance_vectors (torch.Tensor): Distance vectors for a modality. Shape: (batch_size, 1)
        variance_norm_vectors (torch.Tensor): Variance-norm vectors for a modality. Shape: (batch_size, 1)

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Apply softmax to both vectors along the modality dimension
    pd = F.softmax(distance_vectors, dim=1)      # Probability distribution from distances
    ps = F.softmax(variance_norm_vectors, dim=1) # Probability distribution from variances

    # Compute KL Divergence in both directions
    kl_pd_ps = F.kl_div(ps.log(), pd, reduction='batchmean')  # KL(PD || PS)
    kl_ps_pd = F.kl_div(pd.log(), ps, reduction='batchmean')  # KL(PS || PD)

    return kl_pd_ps + kl_ps_pd

def softmax_distributional_matching_loss(distance_vectors_list, variance_norm_vectors_list):
    """
    Computes the combined Calibration and Ordinality Loss for multiple modalities.

    Args:
        distance_vectors_list (list of torch.Tensor): List containing distance vectors for different modalities. Each tensor shape: (batch_size, 1)
        variance_norm_vectors_list (list of torch.Tensor): List containing variance-norm vectors for different modalities. Each tensor shape: (batch_size, 1)

    Returns:
        torch.Tensor: Scalar loss value.
    """
    total_loss = 0.0
    for dv, sv in zip(distance_vectors_list, variance_norm_vectors_list):
        total_loss += calibration_ordinal_loss(dv, sv)
    return total_loss

# Variance Regularization Loss
def variance_regularization_loss(mu, sigma2):
    """
    Computes the KL Divergence between the learned latent distribution and the standard normal distribution.

    Args:
        mu (torch.Tensor): Mean vectors of the latent distributions. Shape: (batch_size, output_size)
        sigma2 (torch.Tensor): Variance vectors of the latent distributions. Shape: (batch_size, output_size)

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # KL(N(mu, sigma^2) || N(0, I)) = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
    kl_div = 0.5 * torch.sum(sigma2 + mu**2 - 1 - torch.log(sigma2 + 1e-8), dim=1)
    return torch.mean(kl_div)

# Total Loss Function
def total_loss_function(L_emo, L_CO, L_regu, 
                        lambda_COV=1e-3, 
                        lambda_COA=1e-3, 
                        lambda_COAV=1e-3, 
                        lambda_R=1e-4):
    """
    Combines Emotion Prediction Loss, Calibration and Ordinality Loss,
    and Variance Regularization Loss into the total loss.

    Args:
        L_emo (torch.Tensor): Emotion prediction loss.
        L_CO (torch.Tensor): Calibration and Ordinality loss.
        L_regu (torch.Tensor): Variance regularization loss.
        lambda_COV (float): Weight for visual-only Calibration and Ordinality loss.
        lambda_COA (float): Weight for audio-only Calibration and Ordinality loss.
        lambda_COAV (float): Weight for crossmodal Calibration and Ordinality loss.
        lambda_R (float): Weight for Variance Regularization loss.

    Returns:
        torch.Tensor: Scalar total loss value.
    """
    return L_emo + lambda_COV * L_CO['COV'] + lambda_COA * L_CO['COA'] + lambda_COAV * L_CO['COAV'] + lambda_R * L_regu
