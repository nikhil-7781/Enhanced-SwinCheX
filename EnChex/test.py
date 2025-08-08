import os
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from config import get_config
from models import build_model
from utils import load_checkpoint
from xai_utils.explainability import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform, class_names):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names
        self.img_paths = self.df['Image Index'].tolist()
        self.labels = [self.encode_labels(lbl) for lbl in self.df['Finding Labels']]

    def encode_labels(self, label_str):
        label_set = set(label_str.split('|'))
        encoded = [1.0 if cls in label_set else 0.0 for cls in self.class_names]
        return torch.tensor(encoded, dtype=torch.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), self.labels[idx], img_path

def get_class_names_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    all_labels = df['Finding Labels'].str.split('|').explode().unique()
    return sorted(all_labels)

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls, _ in tqdm(dataloader, desc="Running inference"):
            imgs = imgs.to(device)
            output = model(imgs)
            logits = output['logits']
            preds.extend(logits.sigmoid().cpu().numpy())
            labels.extend(lbls.numpy())
    return np.array(preds), np.array(labels)

def overlay_cam_on_image(cam, image_tensor):
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
    cam = np.uint8(cam * 255)
    cam = Image.fromarray(cam).resize((image_tensor.shape[2], image_tensor.shape[1]))
    cam = np.array(cam)

    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize
    image_np = np.uint8(image_np * 255)

    plt.imshow(image_np)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    return plt

def generate_gradcam(model, dataloader, output_dir='/content/xai_results', show_inline=False):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    grad_cam = GradCAM(model)

    for img, _, path in tqdm(dataloader, desc="Generating Grad-CAM"):
        img = img[0].unsqueeze(0).to(device)
        img.requires_grad_()

        output = model(img)
        logits = output['logits']
        pred_class = logits.sigmoid().squeeze().argmax().item()
        cam_dict = grad_cam.generate_cam(img, target_class=pred_class)

        cam = cam_dict['cam']
        vis = overlay_cam_on_image(cam, img[0])

        save_path = os.path.join(output_dir, os.path.basename(path[0]))
        vis.savefig(save_path, bbox_inches='tight', pad_inches=0)
        vis.close()

        if show_inline:
            vis.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Path to config file')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--csv_path', default='/content/test.csv', help='Path to test.csv')
    parser.add_argument('--test_dir', default='/content/test', help='Directory with test images')
    parser.add_argument('--show_xai', action='store_true', help='Show Grad-CAM images inline (Colab)')
    args = parser.parse_args()

    config = get_config(args)
    class_names = get_class_names_from_csv(args.csv_path)
    print(f"🩻 Found {len(class_names)} classes: {class_names}")

    transform = transforms.Compose([
        transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = TestImageDataset(args.csv_path, args.test_dir, transform, class_names)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_model(config)
    model.to(device)
    load_checkpoint(model, args.model_path)

    print("🔍 Evaluating model on test set...")
    preds, labels = evaluate(model, dataloader)

    try:
        auc = roc_auc_score(labels, preds, average='macro')
        print("✅ AUC-ROC (macro avg):", auc)
    except Exception as e:
        print("⚠️ ROC AUC could not be calculated:", e)

    print("🎯 Generating Grad-CAM visualizations...")
    generate_gradcam(model, dataloader, show_inline=args.show_xai)

if __name__ == '__main__':
    main()
