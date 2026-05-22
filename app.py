import os
import time

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gradio as gr

from PIL import Image


# =========================================================
# MODEL
# =========================================================

class Attention(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.attn = nn.Sequential(

            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        return x * self.attn(x)



class EmotionCNN(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        # ---------------- Block 1 ----------------

        self.conv1 = nn.Conv2d(
            1,
            32,
            3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv1_2 = nn.Conv2d(
            32,
            32,
            3,
            padding=1
        )

        self.bn1_2 = nn.BatchNorm2d(32)


        # ---------------- Block 2 ----------------

        self.conv2 = nn.Conv2d(
            32,
            64,
            3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.conv2_2 = nn.Conv2d(
            64,
            64,
            3,
            padding=1
        )

        self.bn2_2 = nn.BatchNorm2d(64)


        # ---------------- Block 3 ----------------

        self.conv3 = nn.Conv2d(
            64,
            128,
            3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.conv3_2 = nn.Conv2d(
            128,
            128,
            3,
            padding=1
        )

        self.bn3_2 = nn.BatchNorm2d(128)


        self.pool = nn.MaxPool2d(2, 2)

        self.attn = Attention(128)


        # ---------------- FC ----------------

        self.fc1 = nn.Linear(
            128 * 6 * 6,
            256
        )

        self.drop = nn.Dropout(0.6)

        self.fc2 = nn.Linear(
            256,
            num_classes
        )


    def forward(self, x):

        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = F.relu(
            self.bn1_2(
                self.conv1_2(x)
            )
        )

        x = self.pool(x)


        x = F.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = F.relu(
            self.bn2_2(
                self.conv2_2(x)
            )
        )

        x = self.pool(x)


        x = F.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        x = F.relu(
            self.bn3_2(
                self.conv3_2(x)
            )
        )

        x = self.pool(x)


        x = self.attn(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.drop(x)

        return self.fc2(x)



# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_model.pth"
)

IMG_SIZE = 48

NUM_CLASSES = 7


EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


EMOJI = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"[INFO] Device: {device}")


# =========================================================
# LOAD MODEL
# =========================================================

model = EmotionCNN(
    num_classes=NUM_CLASSES
).to(device)

state = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state)

model.eval()

print("[INFO] Model loaded successfully.")


# =========================================================
# FACE DETECTOR
# =========================================================

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)


def detect_face(gray):

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda f: f[2] * f[3]
    )



# =========================================================
# PREPROCESS
# =========================================================

def preprocess(face):

    face = cv2.resize(
        face,
        (IMG_SIZE, IMG_SIZE)
    )

    face = face.astype(np.float32) / 255.0

    # normalize same as training
    face = (face - 0.5) / 0.5

    tensor = (
        torch.from_numpy(face)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return tensor.to(device)



# =========================================================
# PREDICT
# =========================================================

@torch.no_grad()

def predict(image):

    if image is None:
        return None, "No image uploaded.", None

    start = time.time()

    # PIL -> numpy
    img_rgb = np.array(
        image.convert("RGB")
    )

    img_bgr = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY
    )

    face_box = detect_face(gray)

    if face_box is None:

        face = gray

    else:

        x, y, w, h = face_box

        face = gray[
            y:y+h,
            x:x+w
        ]


    tensor = preprocess(face)

    output = model(tensor)

    probs = F.softmax(
        output,
        dim=1
    )[0]

    probs_np = probs.cpu().numpy()

    pred_idx = int(np.argmax(probs_np))

    emotion = EMOTIONS[pred_idx]

    confidence = probs_np[pred_idx] * 100


    # ---------------- annotate image ----------------

    annotated = img_rgb.copy()

    if face_box is not None:

        color = (0, 255, 0)

        cv2.rectangle(
            annotated,
            (x, y),
            (x+w, y+h),
            color,
            3
        )

        text = f"{emotion} {confidence:.1f}%"

        cv2.putText(
            annotated,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


    elapsed = (
        time.time() - start
    ) * 1000


    result_text = f"""
# {EMOJI[emotion]} {emotion}

### Confidence: {confidence:.2f}%

### Inference Time: {elapsed:.1f} ms
"""


    confidences = {
        EMOTIONS[i]: float(probs_np[i])
        for i in range(len(EMOTIONS))
    }


    return (
        Image.fromarray(annotated),
        result_text,
        confidences
    )



# =========================================================
# UI
# =========================================================

with gr.Blocks(
    title="Emotion Recognition"
) as demo:

    gr.Markdown(
        """
# 🎭 Emotion Recognition System

Upload a face image and the model will predict the emotion.
"""
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Image"
            )

            predict_btn = gr.Button(
                "Analyze Emotion",
                variant="primary"
            )


        with gr.Column():

            output_image = gr.Image(
                label="Detected Face"
            )

            result_output = gr.Markdown()

            output_probs = gr.Label(
                num_top_classes=7,
                label="Emotion Confidence"
            )


    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            output_image,
            result_output,
            output_probs
        ]
    )



# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_error=True
    )
