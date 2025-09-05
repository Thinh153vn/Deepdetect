# 🤖 Project: Faceless - Real-time Deepfake Detection

Faceless là một ứng dụng web được xây dựng bằng Streamlit và PyTorch, sử dụng công nghệ AI Ensemble Learning để phát hiện và phân tích các nội dung deepfake.

Tính năng chính
Phân tích Đa định dạng: Hỗ trợ kiểm tra cả ảnh (JPG, PNG) và video (MP4).

Ensemble Learning: Kết hợp sức mạnh của hai mô hình CNN (EfficientNet) và Transformer (ViT) để tăng độ chính xác.

Phân tích Trực quan: Cung cấp bản đồ nhiệt Grad-CAM và lưới 468 điểm mốc khuôn mặt (MediaPipe).

Phân tích Thời gian thực: Kiểm tra deepfake trực tiếp từ webcam.

Hệ thống Tự cải thiện: Trang Admin cho phép xem xét các trường hợp đáng ngờ và chuẩn bị dữ liệu để huấn luyện lại mô hình.

Hướng dẫn Cài đặt
Clone kho chứa này: git clone ...

Tạo môi trường Conda: conda create -n faceless_env python=3.9

Kích hoạt môi trường: conda activate faceless_env

Cài đặt các thư viện cần thiết: pip install -r requirements.txt

Chạy ứng dụng: streamlit run app_streamlit.py

Công nghệ Sử dụng
Backend & Frontend: Streamlit

AI/ML: PyTorch, Timm, Scikit-learn

Xử lý Ảnh/Video: OpenCV, Pillow, MediaPipe

Trực quan hóa: Plotly Express

---

## 📂 Cấu trúc Thư mục

/Faceless3
|-- app/ # Chứa logic cốt lõi của ứng dụng
| |-- detector.py # Logic phát hiện ensemble và Grad-CAM
| |-- utils.py # Các hàm tiện ích (đọc lịch sử)
| |-- charts.py # Hàm tạo biểu đồ
|-- data/ # Chứa dữ liệu (frames, videos, self-train)
|-- logs/ # Chứa file log lịch sử
|-- models/ # Chứa các file model đã huấn luyện (.h5)
|-- static/ # Chứa file tĩnh (ảnh, video đã tải lên)
|-- templates/ # Chứa các file HTML giao diện
|-- config.py # File cấu hình trung tâm
|-- main.py # File chính để chạy ứng dụng Flask
|-- train.py # Script để huấn luyện model
|-- prepare_dataset.py # Script để phân chia dữ liệu
|-- requirements.txt # Danh sách các thư viện cần thiết
└── README.md # File tài liệu hướng dẫn
