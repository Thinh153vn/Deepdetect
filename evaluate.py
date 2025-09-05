# evaluate.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

# Import các lớp cần thiết từ các file khác
from src.model import DeepfakeDetector, ViTDetector
from src.dataset import FaceDataset # Tái sử dụng lớp Dataset đã viết
import config

def evaluate_model():
    """
    Hàm đánh giá hiệu năng của hệ thống ensemble trên bộ dữ liệu test.
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. Tải các mô hình "vô địch" ---
    print("Loading models...")
    effnet_model = DeepfakeDetector().to(DEVICE)
    effnet_model.load_state_dict(torch.load(config.EFFNET_MODEL_PATH, map_location=DEVICE, weights_only=True))
    effnet_model.eval()

    vit_model = ViTDetector().to(DEVICE)
    vit_model.load_state_dict(torch.load(config.VIT_MODEL_PATH, map_location=DEVICE, weights_only=True))
    vit_model.eval()

    # --- 2. Chuẩn bị Test DataLoader ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = FaceDataset(root_dir='data/test/', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Found {len(test_dataset)} images in test set.")

    # --- 3. Vòng lặp Đánh giá ---
    all_true_labels = []
    all_predicted_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing Progress"):
            images = images.to(DEVICE)
            
            # Lấy dự đoán từ cả hai model
            eff_output = effnet_model(images)
            eff_probs = torch.sigmoid(eff_output)

            vit_output = vit_model(images)
            vit_probs = torch.sigmoid(vit_output)
            
            # Kết hợp kết quả
            final_probs = (eff_probs + vit_probs) / 2.0
            
            # Chuyển xác suất thành nhãn (0 hoặc 1)
            predicted = (final_probs > 0.5).float().cpu().numpy().flatten()
            
            all_true_labels.extend(labels.cpu().numpy().flatten())
            all_predicted_labels.extend(predicted)

    # --- 4. Tính toán và In kết quả ---
    # Nhãn: 0 = REAL, 1 = FAKE
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels)
    recall = recall_score(all_true_labels, all_predicted_labels)
    f1 = f1_score(all_true_labels, all_predicted_labels)
    
    print("\n--- PERFORMANCE REPORT ---")
    print(f"✅ Accuracy: {accuracy:.4f} -> (Tỉ lệ dự đoán đúng tổng thể)")
    print(f"🎯 Precision: {precision:.4f} -> (Trong số các ảnh bị phán là FAKE, bao nhiêu % là FAKE thật)")
    print(f"🔍 Recall: {recall:.4f} -> (Trong tổng số các ảnh FAKE thật, model phát hiện được bao nhiêu %)")
    print(f"📊 F1-Score: {f1:.4f} -> (Chỉ số cân bằng giữa Precision và Recall)")
    print("--------------------------")
    
    # 5. Vẽ Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Lưu hình ảnh
    report_path = 'performance_report.png'
    plt.savefig(report_path)
    print(f"Confusion Matrix saved to {report_path}")
    print("\nDiễn giải Confusion Matrix:")
    print(f" - True REAL (TN): {cm[0][0]} -> (Dự đoán đúng ảnh REAL)")
    print(f" - False FAKE (FP): {cm[0][1]} -> (Dự đoán sai ảnh REAL thành FAKE)")
    print(f" - False REAL (FN): {cm[1][0]} -> (Dự đoán sai ảnh FAKE thành REAL - BỎ LỌT)")
    print(f" - True FAKE (TP): {cm[1][1]} -> (Dự đoán đúng ảnh FAKE)")


if __name__ == '__main__':
    evaluate_model()