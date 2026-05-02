import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import mediapipe as mp

# -----------------------------
# Load External CSS
# -----------------------------
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="EmoVision.ai", layout="wide")

# -----------------------------
# Emotion Labels
# -----------------------------
emotion_labels = [
    "Surprise","Fear","Disgust",
    "Happy","Sad","Angry","Neutral"
]

# -----------------------------
# Load Face Detector (DNN)
# -----------------------------
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# -----------------------------
# MediaPipe Pose
# -----------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# -----------------------------
# Load ViT Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=7
)

model.load_state_dict(torch.load("vit_emotion.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                         [0.5,0.5,0.5])
])

# -----------------------------
# Pose-based Emotion (Fallback)
# -----------------------------
def pose_based_emotion(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark

    left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST].y
    right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
    nose = lm[mp_pose.PoseLandmark.NOSE].y

    # Hands raised
    if left_wrist < nose and right_wrist < nose:
        return "Surprise"

    return "Neutral"

# -----------------------------
# Emotion Prediction (Face + Pose)
# -----------------------------
def predict_emotion(img):

    frame = np.array(img)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame,(300,300)),
        1.0,(300,300),
        (104.0,177.0,123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    face_pred = None

    # Try face detection
    for i in range(detections.shape[2]):

        conf = detections[0,0,i,2]

        if conf > 0.3:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]
            if face.size==0:
                continue

            face = Image.fromarray(face)
            face = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(face)
                pred = torch.argmax(out,1)

            face_pred = emotion_labels[pred.item()]
            break

    # Pose fallback
    pose_pred = pose_based_emotion(frame)

    if face_pred and pose_pred:
        return face_pred + " (Face+Pose)", frame

    if face_pred:
        return face_pred + " (Face)", frame

    if pose_pred:
        return pose_pred + " (Pose)", frame

    return "No human detected", frame

# -----------------------------
# Header
# -----------------------------
st.markdown("<div class='title'>EmoVision.ai</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Face + Pose Based Emotion Recognition</div>", unsafe_allow_html=True)

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns(2)

# -----------------------------
# LEFT PANEL
# -----------------------------
with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Input")

    mode = st.radio("Select Input Type",
                    ["Upload Image","Use Webcam"])

    image = None

    if mode=="Upload Image":
        file = st.file_uploader("Choose Image",
                                type=["jpg","png","jpeg"])
        if file:
            image = Image.open(file).convert("RGB")
            st.image(image,width=350)

    else:
        cam = st.camera_input("Capture Image")
        if cam:
            image = Image.open(cam).convert("RGB")
            st.image(image,width=350)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# RIGHT PANEL
# -----------------------------
with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Result")

    if image is not None:
        if st.button("Start Analysis"):
            emotion, frame = predict_emotion(image)

            st.markdown(f"<div class='result'>{emotion}</div>",
                        unsafe_allow_html=True)

            # Pose Visualization
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            st.image(frame,channels="BGR",
                     caption="Pose Visualization")

    else:
        st.write("Upload image or use webcam")

    st.markdown("</div>", unsafe_allow_html=True)
