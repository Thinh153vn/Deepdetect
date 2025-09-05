# src/model.py
import torch.nn as nn
import timm

# Lớp mô hình 1: EfficientNet (đã có)
class DeepfakeDetector(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(DeepfakeDetector, self).__init__()
        # Tải mô hình EfficientNet-B0 đã được huấn luyện trước
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Lấy số lượng features đầu vào của lớp phân loại cuối cùng
        n_features = self.model.classifier.in_features
        
        # Thay thế lớp phân loại gốc bằng một lớp mới cho bài toán của chúng ta
        self.model.classifier = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x

# Lớp mô hình 2: Vision Transformer (thêm vào)
class ViTDetector(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViTDetector, self).__init__()
        # Tải mô hình ViT đã được huấn luyện trước
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Trong ViT, lớp phân loại cuối cùng thường được gọi là 'head'
        n_features = self.model.head.in_features
        
        # Thay thế lớp cuối cùng cho bài toán của chúng ta
        self.model.head = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x