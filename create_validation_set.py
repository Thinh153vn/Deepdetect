import os
import random
import shutil
from tqdm import tqdm

def create_val_set(train_path, val_path, split_ratio=0.2):
    """
    Tự động tạo tập validation từ tập training.
    """
    print("Creating validation set...")
    
    # Tạo các thư mục val nếu chưa có
    os.makedirs(os.path.join(val_path, 'real'), exist_ok=True)
    os.makedirs(os.path.join(val_path, 'fake'), exist_ok=True)

    for category in ['real', 'fake']:
        src_dir = os.path.join(train_path, category)
        dest_dir = os.path.join(val_path, category)
        
        files = os.listdir(src_dir)
        random.shuffle(files)
        
        split_index = int(len(files) * split_ratio)
        val_files = files[:split_index]
        
        print(f"Moving {len(val_files)} files from train/{category} to val/{category}...")
        for file_name in tqdm(val_files, desc=f"Processing {category}"):
            shutil.move(os.path.join(src_dir, file_name), os.path.join(dest_dir, file_name))
            
    print("Validation set created successfully.")

if __name__ == '__main__':
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    create_val_set(TRAIN_DIR, VAL_DIR)

