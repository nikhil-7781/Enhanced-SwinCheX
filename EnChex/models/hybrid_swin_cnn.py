import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

from models.prototype_layer import PrototypeLayer

class HybridSwinCNN(nn.Module):
    def __init__(self, config, cnn_backbone="resnet50", pretrained=True, num_classes=14):
        super().__init__()
        self.config = config
        self.img_size = config.DATA.IMG_SIZE

        # CNN Backbone (ResNet50)
        if cnn_backbone == "resnet50":
            self.cnn = models.resnet50(pretrained=pretrained)
            self.cnn_features = self.cnn.fc.in_features
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # Remove FC and pooling
        else:
            raise NotImplementedError(f"Backbone {cnn_backbone} not implemented.")

        # Swin Transformer backbone from timm
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True,
            out_indices=(3,)
        )
        # We'll get swin_features dynamically in the first forward pass
        self.fusion = None  # Will initialize after seeing feature shapes

        # Prototype layer (50 prototypes, feature dim 512 after fusion)
        self.prototype_layer = PrototypeLayer(num_prototypes=50, prototype_dim=512)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 + 50, 256),  # 512 from fusion, 50 from prototype similarities
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # For Grad-CAM
        self.cnn_activations = None
        self.swin_activations = None
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def _init_fusion(self, cnn_features, swin_features):
        # Dynamically initialize fusion layer based on input channel sizes
        fusion_in_channels = cnn_features.shape[1] + swin_features.shape[1]
        self.fusion = nn.Conv2d(fusion_in_channels, 512, kernel_size=1).to(cnn_features.device)

    def forward(self, x):
        # CNN path
        cnn_features = self.cnn(x)
        self.cnn_activations = cnn_features

        # Register hook for Grad-CAM
        if cnn_features.requires_grad:
            _ = cnn_features.register_hook(self.activations_hook)

        # Resize the image for Swin if needed
        if x.size(2) != self.img_size or x.size(3) != self.img_size:
            x_swin = F.interpolate(x, size=(self.img_size, self.img_size))
        else:
            x_swin = x

        # Swin Transformer path (timm, features_only=True, out_indices=(3,))
        swin_features_list = self.swin(x_swin)
        if not swin_features_list or not isinstance(swin_features_list, list):
            raise RuntimeError("SwinTransformer returned empty or invalid output. Check input size and model config.")
        swin_features = swin_features_list[0]  # (B, C, H, W)
        self.swin_activations = swin_features

        # Resize to match CNN feature map spatial size
        swin_features = F.interpolate(swin_features, size=cnn_features.shape[2:])

        # Dynamically initialize fusion layer if not done yet
        if self.fusion is None:
            self._init_fusion(cnn_features, swin_features)

        # Fusion
        fused_features = torch.cat([cnn_features, swin_features], dim=1)
        fused_features = self.fusion(fused_features)

        # Prototype comparison
        proto_out = self.prototype_layer(fused_features)
        prototype_similarities = proto_out['similarities']
        prototype_activations = proto_out.get('activation_maps', None)

        # Classification
        pooled_features = self.avgpool(fused_features).flatten(1)
        combined_features = torch.cat([pooled_features, prototype_similarities], dim=1)
        output = self.classifier(combined_features)

        return {
            'logits': output,
            'prototype_similarities': prototype_similarities,
            'prototype_activations': prototype_activations,
            'features': fused_features
        }
