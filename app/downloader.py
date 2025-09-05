import yt_dlp
import requests
import os
import re
import uuid

# Thêm đường dẫn để import config
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def is_image_url(url):
    """Kiểm tra xem URL có trỏ đến một file ảnh không."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    return any(url.lower().endswith(ext) for ext in image_extensions)

def download_content(url):
    """
    Tải nội dung từ một URL, hỗ trợ cả ảnh và video.
    Trả về đường dẫn file đã lưu và loại nội dung ('image' hoặc 'video').
    """
    try:
        if is_image_url(url):
            # Xử lý URL ảnh
            response = requests.get(url, stream=True)
            response.raise_for_status() # Báo lỗi nếu URL không hợp lệ
            
            # Lấy tên file từ URL hoặc tạo tên ngẫu nhiên
            filename = os.path.basename(url.split('?')[0])
            if not filename:
                filename = f"image_{uuid.uuid4().hex[:8]}.jpg"

            filepath = os.path.join(config.UPLOAD_FOLDER, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return filepath, 'image'

        else:
            # Xử lý URL video (YouTube, TikTok, Facebook, etc.)
            ydl_opts = {
                'outtmpl': os.path.join(config.UPLOAD_FOLDER, '%(title)s.%(ext)s'),
                'format': 'best[ext=mp4]/best', # Cố gắng tải file mp4 tốt nhất
                'noplaylist': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
            return filename, 'video'
            
    except Exception as e:
        print(f"Error downloading content from {url}: {e}")
        return None, None
