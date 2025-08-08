import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

class GradCAM:
    def __init__(self, model):
        self.model = model

    def generate_cam(self, input_image, target_class=None):
        """
        Generate CAM for CNN component (multi-label compatible).
        input_image: [B, C, H, W]
        target_class: None, int, or tensor of shape [B]
        """
        self.model.eval()
        output = self.model(input_image)
        logits = output['logits']  # [B, num_classes]

        batch_size, num_classes = logits.shape

        # Determine target class for each sample in the batch
        if target_class is None:
            # For each sample, use the class with the highest logit
            target_class = torch.argmax(logits, dim=1)  # [B]
        elif isinstance(target_class, int):
            # Use the same class for all samples
            target_class = torch.full((batch_size,), target_class, dtype=torch.long, device=logits.device)
        elif isinstance(target_class, torch.Tensor):
            assert target_class.shape == (batch_size,), "target_class tensor must have shape [B]"
        else:
            raise ValueError("target_class must be None, int, or tensor of shape [B]")

        # Zero gradients
        self.model.zero_grad()

        # Create one-hot for each sample
        one_hot = torch.zeros_like(logits)
        one_hot[torch.arange(batch_size), target_class] = 1.0

        # Backward pass for the selected classes
        logits.backward(gradient=one_hot, retain_graph=True)

        # Get CNN Grad-CAM
        cnn_gradients = self.model.gradients  # [B, C, H, W]
        cnn_activations = self.model.cnn_activations  # [B, C, H, W]
        weights = torch.mean(cnn_gradients, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        cnn_cam = torch.sum(weights * cnn_activations, dim=1, keepdim=True)  # [B, 1, H, W]
        cnn_cam = F.relu(cnn_cam)

        # Normalize each CAM in the batch
        cams = []
        for i in range(batch_size):
            cam = cnn_cam[i, 0]
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cams.append(cam.detach().cpu().numpy())
        cnn_cam = np.stack(cams, axis=0)  # [B, H, W]

        # Prepare input images for visualization
        imgs = input_image.detach().cpu().numpy()
        imgs = np.transpose(imgs, (0, 2, 3, 1))  # [B, H, W, C]

        return {
            'cnn_cam': cnn_cam,           # [B, H, W]
            'input_image': imgs,          # [B, H, W, C]
            'prediction': target_class.detach().cpu().numpy(),  # [B]
            'logits': logits.detach().cpu().numpy()             # [B, num_classes]
        }

    def visualize(self, cam_dict, save_path=None, idx=0):
        """
        Visualize the CAM results for one sample in the batch (default: first).
        """
        img = cam_dict['input_image'][idx]
        cam = cam_dict['cnn_cam'][idx]
        pred = cam_dict['prediction'][idx]

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title(f"Original - Pred: {pred}")
        ax[0].axis('off')

        ax[1].imshow(img)
        ax[1].imshow(cam, alpha=0.5, cmap='jet')
        ax[1].set_title("CNN Grad-CAM")
        ax[1].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def visualize_prototypes(model, dataset, num_prototypes=10, save_dir=None):
    """
    Visualize the learned prototypes.
    (Stub for your prototype XAI logic.)
    """
    pass
