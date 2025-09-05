import os

# --- Thư mục ---
MODELS_FOLDER = "models"
UPLOAD_FOLDER = "uploads"
GRADCAM_FOLDER = os.path.join(UPLOAD_FOLDER, "gradcam")
LANDMARK_FOLDER = os.path.join(UPLOAD_FOLDER, "landmarks")
SUSPICIOUS_FOLDER = "suspicious_frames"
HISTORY_FILE = "history.csv"
TRAINING_DATA_FOLDER = "training_data"
TRAINING_FAKE_DIR = os.path.join(TRAINING_DATA_FOLDER, "fake")
TRAINING_REAL_DIR = os.path.join(TRAINING_DATA_FOLDER, "real")


# --- Đường dẫn Model ---
EFFNET_MODEL_PATH = os.path.join(MODELS_FOLDER, 'efficientnet_detector_epoch_10.pth')
VIT_MODEL_PATH = os.path.join(MODELS_FOLDER, 'best_vit_base_patch16_224.pth')

# --- Ngưỡng ---
SELF_TRAIN_THRESHOLD = 95.0 # %

# Tự động tạo các thư mục nếu chúng chưa tồn tại
for folder in [UPLOAD_FOLDER, GRADCAM_FOLDER, LANDMARK_FOLDER, SUSPICIOUS_FOLDER, TRAINING_FAKE_DIR, TRAINING_REAL_DIR]:
    os.makedirs(folder, exist_ok=True)
