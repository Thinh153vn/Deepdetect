# src/predict.py
import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

# Import các lớp mô hình của bạn
from src.model import DeepfakeDetector, ViTDetector

def predict_image(image_path, effnet_model_path, vit_model_path):
    """
    Dự đoán một ảnh sử dụng phương pháp ensembling (kết hợp) hai mô hình.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path not found at {image_path}")
        return

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. Tải các mô hình "vô địch" ---
    print("Loading models...")
    # Tải mô hình EfficientNet
    effnet_model = DeepfakeDetector().to(DEVICE)
    effnet_model.load_state_dict(torch.load(effnet_model_path, map_location=DEVICE, weights_only=True))

    effnet_model.eval() # Chuyển sang chế độ dự đoán

    # Tải mô hình Vision Transformer
    vit_model = ViTDetector().to(DEVICE)
    vit_model.load_state_dict(torch.load(vit_model_path, map_location=DEVICE, weights_only=True))
    vit_model.eval() # Chuyển sang chế độ dự đoán

    # --- 2. Chuẩn bị ảnh đầu vào ---
    # Các phép biến đổi phải giống hệt như lúc training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        processed_image = transform(image).unsqueeze(0).to(DEVICE) # Thêm một chiều cho batch
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # --- 3. Đưa ra dự đoán từ mỗi mô hình ---
    with torch.no_grad(): # Không cần tính gradient khi dự đoán
        # Dự đoán từ EfficientNet
        effnet_output = effnet_model(processed_image)
        effnet_prob = torch.sigmoid(effnet_output).item() # Dùng sigmoid để chuyển thành xác suất (0-1)

        # Dự đoán từ ViT
        vit_output = vit_model(processed_image)
        vit_prob = torch.sigmoid(vit_output).item()

    # --- 4. Kết hợp kết quả (Ensembling) ---
    # Lấy trung bình cộng xác suất của hai mô hình
    final_prob = (effnet_prob + vit_prob) / 2.0
    
    # Đưa ra quyết định cuối cùng
    prediction = "DEEPFAKE" if final_prob > 0.5 else "REAL"

    # In kết quả chi tiết
    print("\n--- BÁO CÁO DỰ ĐOÁN ---")
    print(f"Chuyên gia EfficientNet (CNN) nhận định: {effnet_prob:.2%} là DEEPFAKE")
    print(f"Chuyên gia Vision Transformer (ViT) nhận định: {vit_prob:.2%} là DEEPFAKE")
    print("-----------------------------------")
    print(f"Kết luận của hội đồng (kết hợp): {final_prob:.2%} là DEEPFAKE")
    print(f"==> PHÁN QUYẾT CUỐI CÙNG: {prediction}")
    print("-----------------------------------")

    return prediction, final_prob

if __name__ == '__main__':
    # Tạo đối số dòng lệnh để dễ sử dụng
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference Script")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    # Đường dẫn tới 2 file model "vô địch" của bạn
    # ** NHỚ KIỂM TRA LẠI TÊN FILE NẾU BẠN CHỌN EPOCH KHÁC **
    EFFNET_MODEL_FILE = '../models/deepfake_detector_epoch_10.pth'
    VIT_MODEL_FILE = '../models/vit_detector_epoch_10.pth'
    
    predict_image(args.image, EFFNET_MODEL_FILE, VIT_MODEL_FILE)