import os
import cv2
import csv
import uuid
import pandas as pd

# Thêm đường dẫn để import config
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def log_prediction(data: dict):
    """Ghi lại kết quả dự đoán vào file CSV."""
    file_exists = os.path.isfile(config.HISTORY_FILE)
    
    with open(config.HISTORY_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()  # Viết header nếu file mới được tạo
        writer.writerow(data)

def get_history():
    """Đọc toàn bộ lịch sử dự đoán từ file CSV."""
    if not os.path.exists(config.HISTORY_FILE):
        return []
    try:
        df = pd.read_csv(config.HISTORY_FILE)
        # Sắp xếp để các dự đoán mới nhất hiện lên đầu
        return df.sort_index(ascending=False).to_dict('records')
    except pd.errors.EmptyDataError:
        return [] # Trả về danh sách rỗng nếu file trống

def save_suspicious_frame(frame):
    """Lưu các frame có độ tin cậy deepfake cao để huấn luyện lại sau này."""
    # Tạo một tên file duy nhất
    filename = f"suspicious_{uuid.uuid4().hex[:8]}.jpg"
    save_path = os.path.join(config.SUSPICIOUS_FOLDER, filename)
    cv2.imwrite(save_path, frame)
    print(f"Suspicious frame saved to: {save_path}")

def check_for_notification():
    """Kiểm tra xem có đủ frame đáng ngờ để admin xem xét không."""
    if not os.path.exists(config.SUSPICIOUS_FOLDER):
        return None
        
    suspicious_files = os.listdir(config.SUSPICIOUS_FOLDER)
    num_files = len(suspicious_files)
    
    if num_files >= 50:
        return f"⚠️ Cảnh báo: Có {num_files} frame cần xem xét để huấn luyện lại!"
    return None

def review_suspicious_frame(filename, action):
    """Di chuyển file từ thư mục đáng ngờ sang thư mục training."""
    src_path = os.path.join(config.SUSPICIOUS_FOLDER, filename)
    if not os.path.exists(src_path): return
    
    if action == "confirm_fake":
        dest_path = os.path.join(config.TRAINING_FAKE_DIR, filename)
    elif action == "reject_real":
        dest_path = os.path.join(config.TRAINING_REAL_DIR, filename)
    else: # Xóa file
        os.remove(src_path)
        return
        
    os.rename(src_path, dest_path)
