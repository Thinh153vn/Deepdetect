# src/rename_files.py
import os
from tqdm import tqdm

def sanitize_and_rename(directory_path, prefix):
    """
    Hàm này sẽ đổi tên tất cả các file trong một thư mục theo định dạng:
    prefix_00001.jpg, prefix_00002.jpg, ...
    """
    print(f"Sanitizing files in: {directory_path}")
    
    # Lấy danh sách tất cả các file trong thư mục
    try:
        filenames = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}. Skipping.")
        return

    counter = 1
    # Dùng tqdm để xem tiến trình
    for filename in tqdm(filenames, desc=f"Renaming {prefix} files"):
        # Lấy phần mở rộng của file (ví dụ: .jpg, .png)
        file_extension = os.path.splitext(filename)[1].lower()
        
        # Chỉ xử lý các file ảnh phổ biến
        if file_extension in ['.jpg', '.jpeg', '.png']:
            # Tạo tên file mới
            new_filename = f"{prefix}_{counter:05d}{file_extension}" # :05d -> 00001, 00002,...
            
            # Lấy đường dẫn cũ và mới
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
            
            # Thực hiện đổi tên
            try:
                os.rename(old_path, new_path)
                counter += 1
            except Exception as e:
                print(f"Could not rename {old_path}. Error: {e}")

    print(f"Finished. Renamed {counter - 1} files in {directory_path}.")

if __name__ == '__main__':
    # Đường dẫn tới các thư mục chứa ảnh
    # Giả định script này nằm trong thư mục src/
    TRAIN_REAL_DIR = '../data/train/real/'
    TRAIN_FAKE_DIR = '../data/train/fake/'
    
    # Thực hiện đổi tên cho cả hai thư mục
    sanitize_and_rename(TRAIN_REAL_DIR, 'real')
    sanitize_and_rename(TRAIN_FAKE_DIR, 'fake')

    print("\nAll files have been sanitized.")
    