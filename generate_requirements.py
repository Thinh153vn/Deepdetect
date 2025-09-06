# generate_requirements.py
import os

# Đây là danh sách "chính thức" và duy nhất các thư viện mà dự án cần.
# Chúng ta sẽ quản lý tất cả ở đây.
packages = [
    "streamlit==1.35.0",
    "streamlit-option-menu==0.3.12",
    "torch==2.2.2",
    "torchvision==0.17.2",
    "opencv-python==4.8.0.76",
    "mediapipe==0.10.9",
    "pandas==2.1.4",
    "scikit-learn==1.4.2",
    "tqdm==4.66.4",
    "timm==0.9.16",
    "grad-cam==0.2.1",
    "seaborn==0.13.2",
    "plotly-express==0.4.1",
    "facenet-pytorch" # Thư viện cần cho prepare_data.py
]

# Tên file output
requirements_file = "requirements.txt"

try:
    with open(requirements_file, "w") as f:
        for package in packages:
            f.write(f"{package}\n")
    print(f"Successfully generated '{requirements_file}' with {len(packages)} packages.")
    print("This file is now clean and ready for deployment.")
except Exception as e:
    print(f"An error occurred: {e}")

    
