import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeLayer(nn.Module):
    """
    PrototypeLayer for prototype-based XAI in hybrid CNN-Transformer models.
    Each prototype is a learnable vector in feature space.
    Supports prototype similarity and Grad-CAM-style activation map extraction.
    """
    def __init__(self, num_prototypes, prototype_dim, spatial_dim=1):
        """
        Args:
            num_prototypes (int): Number of prototypes (e.g., 50)
            prototype_dim (int): Feature dimension (e.g., 2048 for ResNet50)
            spatial_dim (int): Size of spatial patch (default 1 for global, >1 for patch-based)
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.spatial_dim = spatial_dim

        # Prototypes: (num_prototypes, prototype_dim)
        self.prototype_vectors = nn.Parameter(
            torch.randn(num_prototypes, prototype_dim)
        )

        # For Grad-CAM: store activations and gradients
        self.last_activations = None
        self.last_grads = None

    def forward(self, x):
        """
        x: (batch_size, feature_dim, H, W)
        Returns:
            similarities: (batch_size, num_prototypes)
            distances: (batch_size, num_prototypes, H*W)
            activation_maps: (batch_size, num_prototypes, H, W)
        """
        batch_size, feature_dim, H, W = x.shape

        # Flatten spatial dimensions
        x_flat = x.view(batch_size, feature_dim, -1)  # (B, C, S)
        S = x_flat.shape[-1]

        # Compute L2 distances between each prototype and each spatial location
        # prototypes: (num_prototypes, feature_dim)
        # x_flat: (B, feature_dim, S)
        prototypes = self.prototype_vectors.unsqueeze(0).unsqueeze(-1)  # (1, P, C, 1)
        x_flat_exp = x_flat.unsqueeze(1)  # (B, 1, C, S)
        # L2 distance: (B, P, S)
        distances = torch.sum((x_flat_exp - prototypes) ** 2, dim=2)

        # Similarity: negative distance
        similarities = -distances  # (B, P, S)

        # For each prototype, take the max similarity over all spatial locations
        max_similarities, max_idx = torch.max(similarities, dim=2)  # (B, P)

        # For XAI: build activation maps for each prototype
        activation_maps = similarities.view(batch_size, self.num_prototypes, H, W)

        # Register hooks for Grad-CAM if needed
        # (Store activations and gradients for visualization)
        activation_maps.requires_grad_(True)
        activation_maps.retain_grad()
        self.last_activations = activation_maps

        def save_gradients(grad):
            self.last_grads = grad
        activation_maps.register_hook(save_gradients)

        return {
            'similarities': max_similarities,   # (B, num_prototypes)
            'distances': distances,             # (B, num_prototypes, S)
            'activation_maps': activation_maps  # (B, num_prototypes, H, W)
        }

    def get_last_gradcam_map(self, prototype_idx=None):
        """
        Returns the last stored activation map and gradients for Grad-CAM visualization.
        If prototype_idx is specified, returns only that prototype's map.
        """
        if self.last_activations is None or self.last_grads is None:
            raise RuntimeError("No activations or gradients stored. Run a backward pass first.")

        # Grad-CAM: weight the activation map by the mean of the gradients
        # shape: (batch, num_prototypes, H, W)
        grads = self.last_grads
        activations = self.last_activations

        # Take mean of gradients over spatial dims (H, W)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (batch, num_prototypes, 1, 1)
        gradcam_map = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (batch, 1, H, W)

        if prototype_idx is not None:
            gradcam_map = gradcam_map[:, prototype_idx:prototype_idx+1, :, :]
        return gradcam_map.detach().cpu()

    def project_prototypes(self, data_loader, feature_extractor):
        """
        (Stub) Update prototypes based on current training data.
        You can implement logic to set each prototype to the feature vector
        of a real patch from the training set that has high activation.
        Args:
            data_loader: DataLoader yielding (images, labels)
            feature_extractor: Function or model to extract features from images
        """
        # Example (not implemented):
        # for images, labels in data_loader:
        #     features = feature_extractor(images)  # (B, C, H, W)
        #     # For each prototype, find patch with highest activation
        #     # Update self.prototype_vectors.data accordingly
        pass
