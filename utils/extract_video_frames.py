import os
import cv2
import uuid

def extract_frames(video_path, output_folder, label, interval=30):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if count % interval == 0:
            filename = f"{label}_{uuid.uuid4().hex[:8]}.jpg"
            path = os.path.join(output_folder, label, filename)
            os.makedirs(os.path.join(output_folder, label), exist_ok=True)
            cv2.imwrite(path, frame)
        count += 1
    cap.release()
    print(f"✅ Extracted frames from {video_path}")

# 🧪 Example usage
# extract_frames("videos/fake_sample.mp4", "data/frames", label="FAKE")
# extract_frames("videos/real_sample.mp4", "data/frames", label="REAL")
