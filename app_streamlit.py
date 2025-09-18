import streamlit as st
import os
import cv2
import pandas as pd
import time
import re
import uuid
import config
import plotly.express as px
import subprocess

# Import các hàm chức năng từ các module đã tạo
from app.detector import (
    predict_image_ensemble,
    predict_video_ensemble,
    process_realtime_frame_fast,
)
from app.utils import (
    get_history,
    log_prediction,
    check_for_notification,
    save_suspicious_frame,
    review_suspicious_frame,
)
from streamlit_option_menu import option_menu


# --- Cấu hình trang ---
st.set_page_config(page_title="Faceless...", page_icon="🤖", layout="wide")


# --- Hàm tiện ích ---
def sanitize_filename(filename):
    """Làm sạch tên file để tránh lỗi."""
    name, ext = os.path.splitext(filename)
    sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
    return (
        f"{sanitized_name}{ext}"
        if sanitized_name
        else f"upload_{uuid.uuid4().hex[:8]}{ext}"
    )


# --- Giao diện Sidebar ---
with st.sidebar:
    selected = option_menu(
        "Faceless Menu",
        ["Home", "Real-time", "History", "Admin", "About"],
        icons=["house", "camera-video", "clock-history", "shield-lock", "info-circle"],
        default_index=0,
    )
    notification = check_for_notification()
    if notification:
        st.sidebar.warning(notification)


# --- Các trang chức năng ---
def page_home():
    st.title("🧠 Deepfake Detection Dashboard")

    if "results" not in st.session_state:
        st.session_state.results = None

    uploaded_file = st.file_uploader(
        "Chọn file ảnh hoặc video", type=["jpg", "png", "mp4"]
    )

    if uploaded_file is not None:
        st.session_state.results = None
        st.info(f"Đã tải lên file: **{uploaded_file.name}**")

        if st.button("🚀 Bắt đầu phân tích", type="primary", use_container_width=True):
            clean_filename = sanitize_filename(uploaded_file.name)
            temp_filepath = os.path.join(config.UPLOAD_FOLDER, clean_filename)
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Đang phân tích, xin chờ trong giây lát..."):
                path_to_save = None
                if uploaded_file.type.startswith("image"):
                    results = predict_image_ensemble(temp_filepath)
                    results["file_type"] = "image"
                    path_to_save = temp_filepath
                else:
                    video_results = predict_video_ensemble(temp_filepath)
                    if video_results:
                        (
                            label, conf, grad_path, eff, vit, frame_path, ann_vid_path,
                        ) = video_results
                        results = {
                            "label": label, "confidence": conf, "gradcam_path": grad_path,
                            "eff_score": eff, "vit_score": vit,
                            "annotated_image_path": ann_vid_path,
                            "file_type": "video",
                        }
                        path_to_save = frame_path
                    else:
                        st.error("Không thể xử lý video.")
                        return

                st.session_state.results = results
                # CẢI TIẾN 3: THAY ĐỔI MÀU THÔNG BÁO
                log_prediction(
                    {
                        "filename": clean_filename,
                        "label": results["label"],
                        "confidence": f"{results['confidence']:.2f}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                st.toast("Đã lưu kết quả vào lịch sử!", icon="✅")

                if (
                    results["label"] == "FAKE"
                    and results["confidence"] >= config.SELF_TRAIN_THRESHOLD
                    and path_to_save
                ):
                    save_suspicious_frame(cv2.imread(path_to_save))
                    st.toast("Đã lưu frame đáng ngờ để huấn luyện lại!", icon="⚠️")

    if st.session_state.results:
        results = st.session_state.results
        st.success("Phân tích hoàn tất!")

        # CẢI TIẾN 1: CÂN ĐỐI BỐ CỤC KẾT QUẢ TRONG 1 TRANG
        st.divider()
        st.subheader("Báo cáo Phân tích Trực quan")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🖼️ File kết quả (đã tích hợp landmark)")
            if results["file_type"] == "image":
                # SỬA LỖI Ở ĐÂY: Bỏ `use_container_width=True`
                st.image(results["annotated_image_path"])
            else:
                if results["annotated_image_path"] and os.path.exists(results["annotated_image_path"]):
                    st.video(results["annotated_image_path"])
                else:
                    st.warning("Không thể tạo video kết quả.")
        with col2:
            st.markdown("##### 🧠 Vùng AI chú ý (Grad-CAM)")
            if results["gradcam_path"] and os.path.exists(results["gradcam_path"]):
                # SỬA LỖI Ở ĐÂY: Bỏ `use_container_width=True`
                st.image(results["gradcam_path"])
            else:
                st.warning("Không có ảnh Grad-CAM.")

        st.divider()
        st.subheader("Phán quyết từ AI")
        if results["label"] == "FAKE":
            st.error(
                f"**Kết quả: {results['label']}** (Độ tin cậy: {results['confidence']:.2f}%)"
            )
        else:
            st.success(
                f"**Kết quả: {results['label']}** (Độ tin cậy: {results['confidence']:.2f}%)"
            )

        with st.expander("Xem chi tiết điểm số từ các mô hình"):
            score_col1, score_col2 = st.columns(2)
            score_col1.metric("EfficientNet Score", f"{results['eff_score']:.2f}%")
            score_col2.metric("ViT Score", f"{results['vit_score']:.2f}%")

def page_realtime():
    st.title("🎥 Real-time Deepfake Detection")
    st.info("Tính năng này yêu cầu quyền truy cập webcam của bạn.")
    
    # CẢI TIẾN 2: TINH GỌN GIAO DIỆN REAL-TIME
    run = st.checkbox("Bật Camera")
    
    if run:
        st.markdown("##### 📷 Webcam Feed")
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret: 
                st.warning("Không thể truy cập camera. Vui lòng thử lại.")
                break
            
            # Chỉ gọi hàm fast, đã bao gồm landmark và kết quả
            display_frame, _, _ = process_realtime_frame_fast(frame)
            
            FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
    else: 
        st.info("Webcam đã tắt.")

def page_history():
    st.title("🗂️ Lịch sử các dự đoán")
    logs = get_history()
    if not logs:
        st.warning("Chưa có dữ liệu trong lịch sử!")
    else:
        df = pd.DataFrame(logs)
        st.subheader("Tỷ lệ dự đoán Real/Fake")
        label_counts = df["label"].value_counts()
        fig = px.pie(values=label_counts.values, names=label_counts.index, title="Tổng quan kết quả",
                     color=label_counts.index, color_discrete_map={"REAL": "green", "FAKE": "red"})
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Tìm kiếm chi tiết")
        search_query = st.text_input("Tìm kiếm theo tên file:")
        if search_query:
            filtered_df = df[df["filename"].str.contains(search_query, case=False, na=False)]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

def page_admin():
    st.title("🔑 Admin Dashboard")

    # CẢI TIẾN 5: ẨN THANH NHẬP MẬT KHẨU SAU KHI ĐĂNG NHẬP
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        password = st.text_input("Enter Admin Password", type="password")
        if password == st.secrets.get("ADMIN_PASSWORD", "admin123"):
            st.session_state.admin_logged_in = True
            st.rerun()
        elif password:
            st.error("Mật khẩu không chính xác.")
    
    if st.session_state.admin_logged_in:
        st.success("Đăng nhập thành công!")
        
        st.subheader("Nâng cấp Mô hình AI")
        num_new_files = len(os.listdir(config.TRAINING_FAKE_DIR)) + len(os.listdir(config.TRAINING_REAL_DIR))
        st.info(f"Hiện có **{num_new_files}** file mới sẵn sàng để huấn luyện.")
        
        if st.button("🚀 Bắt đầu Huấn luyện lại với Dữ liệu Mới", disabled=(num_new_files == 0)):
            try:
                process = subprocess.Popen(["python", "retrain.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                st.info("Quá trình huấn luyện lại đã bắt đầu. Xem tiến trình trong cửa sổ console.")
                st.warning("Sau khi hoàn tất, bạn cần cập nhật file config.py và KHỞI ĐỘNG LẠI ứng dụng.")
            except FileNotFoundError:
                st.error("Lỗi: Không tìm thấy file 'retrain.py'.")

        st.divider()

        st.subheader("🖼️ Các Frame đáng ngờ cần xem xét")
        suspicious_files = os.listdir(config.SUSPICIOUS_FOLDER)
        if not suspicious_files:
            st.info("Hiện không có frame nào cần xem xét.")
        else:
            for filename in suspicious_files:
                filepath = os.path.join(config.SUSPICIOUS_FOLDER, filename)
                with st.container():
                    st.image(filepath, width=300)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Xác nhận là FAKE", key=f"confirm_{filename}", use_container_width=True):
                            review_suspicious_frame(filename, "confirm_fake")
                            st.toast(f"{filename} đã được chuyển vào training set (FAKE).", icon="✅")
                            st.rerun()
                    with col2:
                        if st.button("❌ Bác bỏ (là REAL)", key=f"reject_{filename}", use_container_width=True):
                            review_suspicious_frame(filename, "reject_real")
                            st.toast(f"{filename} đã được chuyển vào training set (REAL).", icon="❌")
                            st.rerun()
                    st.divider()

def page_about():
    # CẢI TIẾN 4: CẬP NHẬT NỘI DUNG TAB ABOUT
    st.title("ℹ️ Giới thiệu về Dự án 'Faceless'")
    st.markdown("""
    **Faceless** là một dự án demo nhằm xây dựng một công cụ phát hiện deepfake mạnh mẽ và trực quan, 
    giúp người dùng dễ dàng xác thực tính chân thực của nội dung số.

    ### Công nghệ Cốt lõi
    Ứng dụng được xây dựng trên một nền tảng công nghệ AI hiện đại:
    - **Ensemble Learning:** Kết hợp sức mạnh của hai kiến trúc mô hình hàng đầu:
        - **EfficientNet (CNN):** Tập trung vào việc phân tích các chi tiết nhỏ, kết cấu và các sai sót ở cấp độ pixel.
        - **Vision Transformer (ViT):** Phân tích các mối quan hệ và sự nhất quán trên toàn bộ khuôn mặt.
    - **Explainable AI (XAI):**
        - **Grad-CAM:** Hiển thị bản đồ nhiệt, cho biết những vùng nào trên khuôn mặt mà AI chú ý nhất để đưa ra quyết định.
    - **Facial Analysis:**
        - **MediaPipe:** Sử dụng lưới 468 điểm mốc để phân tích và trực quan hóa cấu trúc khuôn mặt, hoạt động hiệu quả trên nhiều góc độ.

    ### Cách hoạt động
    1.  **Đầu vào:** Người dùng tải lên ảnh hoặc video.
    2.  **Phân tích song song:** Cả hai mô hình AI (EfficientNet và ViT) cùng lúc phân tích nội dung.
    3.  **Tổng hợp kết quả:** Các điểm số được kết hợp để đưa ra phán quyết cuối cùng với độ tin cậy cao hơn.
    4.  **Trực quan hóa:** Các kết quả phân tích sâu như Grad-CAM và Landmark được tạo ra và hiển thị cho người dùng.
    
    ### Tuyên bố miễn trừ trách nhiệm
    Đây là một sản phẩm phục vụ mục đích học tập và demo. Kết quả do AI đưa ra chỉ mang tính tham khảo và không đảm bảo chính xác 100%.
    """)

# --- Điều hướng trang ---
if selected == "Home":
    page_home()
elif selected == "Real-time":
    page_realtime()
elif selected == "History":
    page_history()
elif selected == "Admin":
    page_admin()
elif selected == "About":
    page_about()