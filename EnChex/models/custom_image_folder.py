import torch
import torchvision
from torchvision.datasets import DatasetFolder, ImageFolder, VisionDataset
from PIL import Image
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    return filename.lower().endswith(extensions)

def is_image_file(filename: str) -> bool:
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
        is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances

class MyDatasetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        csv_path: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None
    ) -> None:
        super(MyDatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        # samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        samples = self.make_custom_dataset(self.root, csv_path)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_custom_dataset(
        directory: str,
        csv_path: str,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Loads image paths and multi-label, multi-hot encoded targets from the given CSV.
        CSV must have columns: 'Image Index' and 'Finding Labels' (pipe-separated for multi-label).
        """
        df = pd.read_csv(csv_path)
        # Ensure correct columns
        if 'Image Index' not in df.columns or 'Finding Labels' not in df.columns:
            raise ValueError("CSV must contain 'Image Index' and 'Finding Labels' columns.")

        image_paths = [os.path.join(directory, fname) for fname in df['Image Index'].values]

        # Split finding labels by '|' and handle multi-label
        multilabels = [str(labels).split('|') if pd.notna(labels) else ['No Finding'] for labels in df['Finding Labels']]

        # Fit MultiLabelBinarizer on all possible classes
        all_classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
            'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        mlb = MultiLabelBinarizer(classes=all_classes)
        binary_labels = mlb.fit_transform(multilabels)

        # Each sample: (image_path, multi-hot tensor)
        return [(img_path, torch.tensor(label, dtype=torch.float32)) for img_path, label in zip(image_paths, binary_labels)]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class MyImageFolder(MyDatasetFolder):
    def __init__(
        self,
        root: str,
        csv_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(MyImageFolder, self).__init__(root, csv_path, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                            transform=transform,
                                            target_transform=target_transform,
                                            is_valid_file=is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
