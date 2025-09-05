import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
import mediapipe as mp
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Thêm đường dẫn của thư mục gốc để Python có thể tìm thấy module 'src' và 'config'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import DeepfakeDetector, ViTDetector
import config

# --- Tải các model và công cụ một lần duy nhất khi khởi động ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading AI models...")
EFFNET_MODEL = DeepfakeDetector().to(DEVICE)
EFFNET_MODEL.load_state_dict(torch.load(config.EFFNET_MODEL_PATH, map_location=DEVICE, weights_only=True))
EFFNET_MODEL.eval()

VIT_MODEL = ViTDetector().to(DEVICE)
VIT_MODEL.load_state_dict(torch.load(config.VIT_MODEL_PATH, map_location=DEVICE, weights_only=True))
VIT_MODEL.eval()
print("AI models loaded successfully.")

print("Initializing MediaPipe tools...")
MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
MP_DRAWING = mp.solutions.drawing_utils
DRAWING_SPEC = MP_DRAWING.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
print("MediaPipe tools initialized.")


# --- Các hàm chức năng ---

def predict_image_ensemble(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_pil = Image.open(image_path).convert("RGB")
    tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        eff_output = EFFNET_MODEL(tensor)
        eff_prob = torch.sigmoid(eff_output).item()
        vit_output = VIT_MODEL(tensor)
        vit_prob = torch.sigmoid(vit_output).item()
    
    final_prob = (eff_prob + vit_prob) / 2.0
    label = "FAKE" if final_prob > 0.5 else "REAL"
    confidence = final_prob * 100 if label == "FAKE" else (1 - final_prob) * 100

    target_layers = [EFFNET_MODEL.model.conv_head]
    cam = GradCAM(model=EFFNET_MODEL, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
    
    rgb_img_for_cam = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img_for_cam = cv2.resize(rgb_img_for_cam, (224, 224))
    rgb_img_for_cam = np.float32(rgb_img_for_cam) / 255
    visualization = show_cam_on_image(rgb_img_for_cam, grayscale_cam, use_rgb=True)
    
    visualization_uint8 = (visualization * 255).astype(np.uint8)
    image_to_save = cv2.cvtColor(visualization_uint8, cv2.COLOR_RGB2BGR)

    gradcam_filename = "gradcam_" + os.path.basename(image_path)
    gradcam_path = os.path.join(config.GRADCAM_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, image_to_save)

    annotated_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    results_mp = MP_FACE_MESH.process(image_rgb)
    if results_mp.multi_face_landmarks:
        for face_landmarks in results_mp.multi_face_landmarks:
            MP_DRAWING.draw_landmarks(
                image=annotated_image, landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=DRAWING_SPEC)

    color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
    text = f"{label} ({confidence:.2f}%)"
    cv2.putText(annotated_image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    annotated_filename = "annotated_" + os.path.basename(image_path)
    annotated_image_path = os.path.join(config.UPLOAD_FOLDER, annotated_filename)
    cv2.imwrite(annotated_image_path, annotated_image)
    
    return {
        "label": label, "confidence": confidence, "gradcam_path": gradcam_path,
        "eff_score": eff_prob * 100, "vit_score": vit_prob * 100,
        "annotated_image_path": annotated_image_path
    }


def predict_video_ensemble(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None, None, None, None, None, None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60: fps = 30
    
    frames_to_process_indices = [i for i in range(0, frame_count, int(fps))]
    if not frames_to_process_indices: frames_to_process_indices = [0]

    predictions, highest_fake_score, best_frame_for_analysis = [], -1, None
    
    for frame_idx in frames_to_process_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        temp_frame_path = os.path.join(config.UPLOAD_FOLDER, f"temp_frame_{frame_idx}.jpg")
        cv2.imwrite(temp_frame_path, frame)
        result = predict_image_ensemble(temp_frame_path)
        predictions.append(result)
        
        current_fake_score = (result['eff_score'] + result['vit_score']) / 2.0
        if result['label'] == 'FAKE' and current_fake_score > highest_fake_score:
            highest_fake_score = current_fake_score
            best_frame_for_analysis = temp_frame_path
    
    if not predictions:
        cap.release()
        return None, None, None, None, None, None, None

    fake_count = sum(1 for p in predictions if p['label'] == 'FAKE')
    video_label = "FAKE" if fake_count / len(predictions) > 0.4 else "REAL"

    if video_label == "FAKE":
        worst_frame_result = max(predictions, key=lambda x: (x['eff_score'] + x['vit_score']) / 2.0)
        final_confidence, gradcam_path, eff_score, vit_score, frame_path_for_landmarks = (
            worst_frame_result['confidence'], worst_frame_result['gradcam_path'],
            worst_frame_result['eff_score'], worst_frame_result['vit_score'], best_frame_for_analysis)
    else:
        avg_confidence = 100 - (sum([(p['eff_score'] + p['vit_score']) / 2.0 for p in predictions]) / len(predictions))
        final_confidence, gradcam_path, eff_score, vit_score, frame_path_for_landmarks = (
            avg_confidence, predictions[0]['gradcam_path'],
            sum([p['eff_score'] for p in predictions]) / len(predictions),
            sum([p['vit_score'] for p in predictions]) / len(predictions), None)

    annotated_video_filename = "annotated_" + os.path.basename(video_path)
    annotated_video_path = os.path.join(config.UPLOAD_FOLDER, annotated_video_filename)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open VideoWriter. Trying with 'mp4v' as a fallback.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("Fatal Error: Could not open VideoWriter with any codec.")
            cap.release()
            return video_label, final_confidence, gradcam_path, eff_score, vit_score, frame_path_for_landmarks, None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = MP_FACE_MESH.process(image_rgb)
        if results_mp.multi_face_landmarks:
            for face_landmarks in results_mp.multi_face_landmarks:
                MP_DRAWING.draw_landmarks(
                    image=frame, landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=DRAWING_SPEC)
        color = (0, 0, 255) if video_label == "FAKE" else (0, 255, 0)
        text = f"{video_label} (Confidence: {final_confidence:.2f}%)"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        out.write(frame)

    cap.release()
    out.release()
            
    return video_label, final_confidence, gradcam_path, eff_score, vit_score, frame_path_for_landmarks, annotated_video_path

def process_realtime_frame_fast(frame):
    """
    Phân tích một frame từ webcam, bao gồm cả deepfake detection và facial landmarks.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = EFFNET_MODEL(tensor)
        prob = torch.sigmoid(output).item()
    label = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob * 100 if label == "FAKE" else (1 - prob) * 100

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = MP_FACE_MESH.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            MP_DRAWING.draw_landmarks(
                image=frame, landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=DRAWING_SPEC)
    
    color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
    text = f"{label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame, label, confidence

