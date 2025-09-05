# src/dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Lấy đường dẫn và nhãn cho ảnh REAL (nhãn 0)
        real_path = os.path.join(root_dir, 'real')
        for img_name in os.listdir(real_path):
            self.image_paths.append(os.path.join(real_path, img_name))
            self.labels.append(0)

        # Lấy đường dẫn và nhãn cho ảnh FAKE (nhãn 1)
        fake_path = os.path.join(root_dir, 'fake')
        for img_name in os.listdir(fake_path):
            self.image_paths.append(os.path.join(fake_path, img_name))
            self.labels.append(1)

    def __len__(self):
        # Trả về tổng số lượng ảnh
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Lấy một ảnh và nhãn tại một vị trí (index) cụ thể
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, float(label)