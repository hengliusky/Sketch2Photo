import numpy as np
from PIL import Image, ImageDraw, ImageChops
from skimage.color import rgb2lab, lab2rgb
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 256


class MyDataset(Dataset):
    def __init__(self, real_paths=None, sketch_paths=None):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
        self.size = SIZE
        self.real_paths = real_paths
        self.sketch_paths = sketch_paths
        self.resize = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

    def __getitem__(self, idx):
        ### input sketch
        sketch = Image.open(self.sketch_paths[idx]).convert("RGB")
        ### input real
        real = Image.open(self.real_paths[idx]).convert("RGB")
        sketch = self.resize(sketch)
        real = self.resize(real)
        sketch = self.transforms(sketch)
        real = self.transforms(real)
        return {'sketch': sketch, 'real': real}

    def __len__(self):
        return len(self.sketch_paths)
