import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

import os
import glob
from PIL import Image
import torchvision.transforms as transforms

def count_occupied_cell_classes(base_path, S=7, num_classes=5, splits=("train", "val")):
    """Count class frequencies the way ``AccidentDetectionLoss`` sees them.

    Classification loss runs only on *occupied grid cells*, and ``TrafficGridDataset``
    keeps one object per cell (first annotation wins). So we replicate that exact
    bucketing here rather than counting raw label lines -- otherwise the derived
    weights would be skewed by objects that the loss never actually scores.

    Returns a ``np.int64`` array of length ``num_classes`` (occupied-cell counts).
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for split in splits:
        label_dir = os.path.join(str(base_path), "labels", split)
        if not os.path.isdir(label_dir):
            continue
        for label_path in glob.glob(os.path.join(label_dir, "*.txt")):
            occupied = {}  # (grid_y, grid_x) -> class_idx, first object per cell wins
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    x_center, y_center = float(parts[1]), float(parts[2])
                    grid_x = min(max(int(x_center * S), 0), S - 1)
                    grid_y = min(max(int(y_center * S), 0), S - 1)
                    if (grid_y, grid_x) not in occupied:
                        occupied[(grid_y, grid_x)] = cls
            for cls in occupied.values():
                if 0 <= cls < num_classes:
                    counts[cls] += 1
    return counts


def effective_number_weights(counts, beta=0.999, max_ratio=10.0, class0_cap=None):
    """Class-balanced "effective number" weights (Cui et al., 2019).

    ``w_c = (1 - beta) / (1 - beta**n_c)``, then mean-normalized to 1 so the overall
    classification-loss scale is unchanged. ``class0_cap`` clamps the "No accident"
    weight (analysis recommendation #2); ``max_ratio`` caps the spread so a rare class
    can't dominate the gradient. Classes absent from occupied cells get the mean weight.

    Returns a plain ``list[float]`` of length ``len(counts)``.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = len(counts)
    seen = counts > 0

    weights = np.ones(n, dtype=np.float64)
    if seen.any():
        eff_num = 1.0 - np.power(beta, counts[seen])
        weights[seen] = (1.0 - beta) / eff_num
        weights[~seen] = weights[seen].mean()  # unseen class -> neutral weight
        weights = weights / weights.mean()      # mean-normalize to 1

    if class0_cap is not None and n > 0:
        weights[0] = min(weights[0], float(class0_cap))

    if max_ratio is not None:
        floor = weights[weights > 0].min() if (weights > 0).any() else 1.0
        weights = np.minimum(weights, floor * float(max_ratio))

    return weights.tolist()


class BaseDataset(Dataset):
    def __init__(self, dataset: TensorDataset, train_ratio=0.7, val_ratio=0.2, batch_size=32):
        super().__init__()

        N = len(dataset)
        n_train = int(train_ratio * N)
        n_val = int(val_ratio * N)
        n_test = N - n_train - n_val

        self.train_set, self.val_set, self.test_set = random_split(
            dataset, [n_train, n_val, n_test]
        )

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_set, batch_size=batch_size//2, shuffle=False)
        self.test_loader  = DataLoader(self.test_set, batch_size=batch_size//2, shuffle=False)
    
    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x

class TrafficGridDataset(Dataset):
    def __init__(self, base_path, split="train", S=7, num_classes=5, img_size=224):
        """
        split: "train" or "val" to navigate the directory structure.
        """
        self.S = S
        self.num_classes = num_classes
        
        # Resolve directories based on your layout
        self.img_dir = os.path.join(base_path, "images", split)
        self.label_dir = os.path.join(base_path, "labels", split)
        
        # Grab all image files (handles jpg, jpeg, png)
        self.img_paths = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg")) + 
            glob.glob(os.path.join(self.img_dir, "*.jpeg")) + 
            glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        
        # Photometric augmentation on the training split only (val/test stay
        # deterministic). These augs don't move object positions, so the grid/bbox
        # targets remain valid. Geometric augs (flips/crops) would require
        # transforming the targets too, so they are intentionally left out here.
        aug = []
        if split == "train":
            aug = [
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
            ]

        # Basic transformations to scale and tensorize images
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load and process the image
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # 2. Find matching label file name (same base name, but .txt)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        
        # Initialize an empty target matrix: [7, 7, 10]
        # Channels: 1 (Objectness) + 5 (Classes) + 4 (BBox Coords) = 10
        target = torch.zeros((self.S, self.S, 1 + self.num_classes + 4))
        
        # If label file exists and isn't empty, read annotations
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, "r") as f:
                for line in f:
                    # Assumes standard YOLO file format: class_idx x_center y_center width height
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                        
                    severity_idx = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # Calculate which grid cell row/col this accident center falls into
                    grid_x = int(x_center * self.S)
                    grid_y = int(y_center * self.S)
                    
                    # Ensure coordinates are within grid boundaries
                    grid_x = min(max(grid_x, 0), self.S - 1)
                    grid_y = min(max(grid_y, 0), self.S - 1)
                    
                    # Keep one object per cell: the `== 0` guard skips cells that are
                    # already occupied, but every annotation in the file is processed
                    # (do NOT break here, or only the first object in the image is kept).
                    if target[grid_y, grid_x, 0] == 0:
                        target[grid_y, grid_x, 0] = 1.0  # Objectness = 1
                        target[grid_y, grid_x, 1 + severity_idx] = 1.0  # One-hot class
                        target[grid_y, grid_x, 6:10] = torch.tensor([x_center, y_center, w, h])

        return image, target

class TrafficDataManager:
    def __init__(self, base_path, batch_size=32, S=7, num_classes=5):
        """
        Builds distinct datasets from your disk partition folders
        and exposes clean DataLoader iterators.
        """
        self.train_set = TrafficGridDataset(
            base_path=base_path, split="train", S=S, num_classes=num_classes
        )
        
        self.val_set = TrafficGridDataset(
            base_path=base_path, split="val", S=S, num_classes=num_classes
        )
        
        val_len = len(self.val_set)
        n_val = val_len // 2
        n_test = val_len - n_val
        
        self.val_subset, self.test_subset = random_split(
            self.val_set, [n_val, n_test], generator=torch.Generator().manual_seed(42)
        )

        # Build data loaders matching your layout parameters
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_subset, batch_size=batch_size // 2, shuffle=False)
        self.test_loader  = DataLoader(self.test_subset, batch_size=1, shuffle=False)
    
    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def __len__(self):
        return len(self.train_set) + len(self.val_set)