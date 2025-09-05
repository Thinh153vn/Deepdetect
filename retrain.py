import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import shutil

# Import các lớp cần thiết
from src.model import DeepfakeDetector, ViTDetector
from src.dataset import FaceDataset
import config

def merge_new_data():
    """Hợp nhất dữ liệu đã được admin xem xét vào bộ dữ liệu training chính."""
    print("Merging new data into training set...")
    
    # Copy các file FAKE mới
    for filename in os.listdir(config.TRAINING_FAKE_DIR):
        src = os.path.join(config.TRAINING_FAKE_DIR, filename)
        dst = os.path.join("data/train/fake", filename)
        shutil.move(src, dst) # Dùng move để thư mục training_data trống sau khi hợp nhất

    # Copy các file REAL mới
    for filename in os.listdir(config.TRAINING_REAL_DIR):
        src = os.path.join(config.TRAINING_REAL_DIR, filename)
        dst = os.path.join("data/train/real", filename)
        shutil.move(src, dst)
        
    print("Data merging complete.")

def run_retraining():
    """
    Chạy quá trình fine-tuning cho cả hai mô hình.
    """
    # --- 1. Hợp nhất dữ liệu mới ---
    merge_new_data()

    # --- 2. Cấu hình chung ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 3e-5 # Dùng learning rate nhỏ hơn cho fine-tuning
    BATCH_SIZE = 32
    NUM_EPOCHS_FINETUNE = 10 # Chỉ cần tinh chỉnh trong vài epochs
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = FaceDataset(root_dir='data/train/', transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- 3. Fine-tune từng mô hình ---
    models_to_retrain = {
        "efficientnet": (DeepfakeDetector, config.EFFNET_MODEL_PATH, "efficientnet_detector_v2.pth"),
        "vit": (ViTDetector, config.VIT_MODEL_PATH, "vit_detector_v2.pth")
    }

    for name, (model_class, old_path, new_name) in models_to_retrain.items():
        print(f"\n--- Starting fine-tuning for {name} ---")
        
        # TẢI LẠI MODEL CŨ
        model = model_class().to(DEVICE)
        model.load_state_dict(torch.load(old_path, map_location=DEVICE, weights_only=True))
        
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(NUM_EPOCHS_FINETUNE):
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE}")
            for images, labels in loop:
                images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())

        # LƯU RA MODEL MỚI VỚI TÊN KHÁC
        new_model_path = os.path.join(config.MODELS_FOLDER, new_name)
        torch.save(model.state_dict(), new_model_path)
        print(f"Fine-tuning complete. New model saved to: {new_model_path}")

if __name__ == '__main__':
    run_retraining()