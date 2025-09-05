import os
import csv
import shutil

# Cấu hình
LOG_PATH = 'logs/predictions_log.csv'
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_DIR = 'data/self_train'
THRESHOLD = 80  # Confidence tối thiểu để chấp nhận

# Tạo thư mục đích nếu chưa có
for label in ['real', 'fake']:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# Xử lý từng dòng log
with open(LOG_PATH, 'r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        try:
            confidence = float(row['confidence'])
            label = row['label'].lower()
            file_type = row['type']
            source = row['source']

            if confidence < THRESHOLD or file_type not in ['upload', 'webcam']:
                continue

            src_path = os.path.join(UPLOAD_FOLDER, source)
            if not os.path.exists(src_path):
                continue

            dst_path = os.path.join(OUTPUT_DIR, label, source)
            shutil.copy2(src_path, dst_path)
            print(f"✅ Copied {source} → {label}")

        except Exception as e:
            print(f"❌ Skipping row due to error: {e}")
