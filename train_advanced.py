import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import argparse

# Import các lớp cần thiết
from src.model import DeepfakeDetector, ViTDetector
from src.dataset import FaceDataset
import config

def train_model(model_name, num_epochs):
    """
    Hàm chính để huấn luyện một mô hình với tập validation.
    """
    # --- 1. Cấu hình ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    
    # --- 2. Chuẩn bị Dữ liệu ---
    # Áp dụng Data Augmentation cho tập train
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Không áp dụng Augmentation cho tập validation
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = FaceDataset(root_dir='data/train/', transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    val_dataset = FaceDataset(root_dir='data/val/', transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # --- 3. Khởi tạo Mô hình ---
    print(f"Initializing model: {model_name}")
    if 'efficientnet' in model_name:
        model = DeepfakeDetector(model_name=model_name).to(DEVICE)
    elif 'vit' in model_name:
        model = ViTDetector(model_name=model_name).to(DEVICE)
    else:
        raise ValueError("Model name not recognized. Use 'efficientnet_b0' or 'vit_base_patch16_224'.")
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Vòng lặp Huấn luyện & Đánh giá ---
    best_val_accuracy = 0.0
    best_model_path = os.path.join(config.MODELS_FOLDER, f"best_{model_name}.pth")

    for epoch in range(num_epochs):
        # -- Training Phase --
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for images, labels in train_loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)
        
        val_accuracy = correct_predictions / total_samples
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  - Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  - Avg Val Loss: {avg_val_loss:.4f}")
        print(f"  - Val Accuracy: {val_accuracy:.4f}")
        
        # -- Save Best Model --
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved with accuracy: {best_val_accuracy:.4f}")
    
    print(f"\nTraining complete. The best model was saved to {best_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced Training Script for Deepfake Detection")
    parser.add_argument('--model', type=str, required=True, help="Model to train (e.g., 'efficientnet_b0', 'vit_base_patch16_224')")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs to train")
    args = parser.parse_args()
    
    train_model(args.model, args.epochs)