import torch
import cv2
import os
from facenet_pytorch import MTCNN
from tqdm import tqdm

# --- Cấu hình ---
# Đường dẫn đến thư mục chứa video gốc (đã được chia thành real/fake)
RAW_DATASET_DIR = 'dataset_raw'
# Đường dẫn đến thư mục để lưu các ảnh khuôn mặt đã xử lý
PROCESSED_DATASET_DIR = 'data/train'
# Chỉ xử lý 1 frame mỗi N frame để tránh dữ liệu bị trùng lặp quá nhiều
FRAME_SAMPLING_RATE = 15 
# Kích thước tối thiểu của khuôn mặt (tính bằng pixel) để được chấp nhận
MIN_FACE_SIZE = 40 

def prepare_dataset():
    """
    Hàm chính để quét video, trích xuất khuôn mặt và lưu lại.
    """
    # Khởi tạo MTCNN để phát hiện khuôn mặt
    # Sử dụng GPU nếu có để tăng tốc độ
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # keep_all=False: chỉ giữ lại khuôn mặt có độ tin cậy cao nhất
    # post_process=False: không cần căn chỉnh ảnh, chỉ cần cắt
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=MIN_FACE_SIZE, device=device)

    # Tạo các thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.join(PROCESSED_DATASET_DIR, 'real'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATASET_DIR, 'fake'), exist_ok=True)

    # Xử lý tuần tự các loại video
    for category in ['real', 'fake']:
        print(f"\nProcessing '{category}' videos...")
        source_dir = os.path.join(RAW_DATASET_DIR, category)
        output_dir = os.path.join(PROCESSED_DATASET_DIR, category)

        if not os.path.isdir(source_dir):
            print(f"Warning: Directory not found, skipping: {source_dir}")
            continue

        video_files = [f for f in os.listdir(source_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        # Dùng tqdm để tạo thanh tiến trình
        for video_file in tqdm(video_files, desc=f"Processing {category} videos"):
            video_path = os.path.join(source_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Warning: Could not open video file {video_path}")
                continue

            frame_count = 0
            saved_face_count = 0
            video_name = os.path.splitext(video_file)[0] # Lấy tên video không có phần mở rộng

            while True:
                ret, frame = cap.read()
                if not ret:
                    break # Kết thúc video

                # Chỉ xử lý frame theo tỉ lệ đã định
                if frame_count % FRAME_SAMPLING_RATE == 0:
                    # MTCNN cần ảnh ở định dạng RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Phát hiện khuôn mặt
                    boxes, _ = mtcnn.detect(frame_rgb)
                    
                    if boxes is not None:
                        # Lấy tọa độ bounding box
                        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                        
                        # Cắt khuôn mặt từ frame gốc (BGR)
                        face = frame[y1:y2, x1:x2]
                        
                        # Kiểm tra xem khuôn mặt có hợp lệ không (đủ lớn)
                        if face.size > 0:
                            # Lưu ảnh khuôn mặt
                            save_path = os.path.join(output_dir, f'{video_name}_frame{frame_count}.jpg')
                            cv2.imwrite(save_path, face)
                            saved_face_count += 1
                
                frame_count += 1
            
            cap.release()
            # print(f"  -> Saved {saved_face_count} faces from {video_file}")

    print("\nData preparation complete!")

if __name__ == '__main__':
    prepare_dataset()
