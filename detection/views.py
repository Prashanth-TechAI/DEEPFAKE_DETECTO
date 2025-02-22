import os
import tempfile
import json
import numpy as np
import cv2
from PIL import Image

# Audio detection libraries
import librosa
from tensorflow.keras.models import load_model

# Video detection libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

from drf_yasg.utils import swagger_auto_schema
from .serializers import AudioUploadSerializer, VideoUploadSerializer

# -------------------------
# Audio Detection Function
# -------------------------
def detect_fake_audio_from_file(audio_file, model_path):
    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    try:
        audio, sr = librosa.load(tmp_path, sr=None)
        n_mels = 128
        spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        spect_db = librosa.power_to_db(spect, ref=np.max)
        expected_frames = 109
        if spect_db.shape[1] < expected_frames:
            pad_width = expected_frames - spect_db.shape[1]
            spect_db = np.pad(spect_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            spect_db = spect_db[:, :expected_frames]
        features = spect_db[..., np.newaxis]
        features = np.expand_dims(features, axis=0)
        model = load_model(model_path)
        predictions = model.predict(features)
        fake_probability = predictions[0][0]
        fake_percentage = float(fake_probability) * 100
        if fake_percentage >= 85:
            status_str = "Highly Fake"
        elif fake_percentage >= 70:
            status_str = "Fake"
        elif fake_percentage >= 50:
            status_str = "Possibly Fake"
        else:
            status_str = "Genuine"
        result = {
            "result": {
                "fake_percentage": round(fake_percentage, 2),
                "status": status_str
            }
        }
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)

# -------------------------
# Video Detection Model Architecture
# -------------------------
class VideoDetectionModel(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(VideoDetectionModel, self).__init__()
        base_model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional, batch_first=True)
        self.dp = nn.Dropout(0.4)
        fc_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.linear1 = nn.Linear(fc_input_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        logits = self.dp(self.linear1(x))
        return logits

# -------------------------
# Video Detection Function
# -------------------------
def detect_fake_video_from_file(video_file, model_path):
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name
    try:
        im_size = 112
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        video_transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return {"error": "Video has no frames."}
        num_frames = 10
        frame_interval = max(total_frames // num_frames, 1)
        frames = []
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) < num_frames:
            return {"error": "Could not extract enough frames from video."}
        transformed_frames = []
        for frame in frames:
            pil_img = Image.fromarray(frame)
            transformed_img = video_transform(pil_img)
            transformed_frames.append(transformed_img)
        video_tensor = torch.stack(transformed_frames).unsqueeze(0)  # (1, num_frames, 3, 112, 112)
        model = VideoDetectionModel(num_classes=2)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        with torch.no_grad():
            logits = model(video_tensor)
            probs = torch.softmax(logits, dim=1)
            fake_probability = probs[0][0].item()
            fake_percentage = fake_probability * 100
        if fake_percentage >= 85:
            status_str = "Highly Fake"
        elif fake_percentage >= 70:
            status_str = "Fake"
        elif fake_percentage >= 50:
            status_str = "Possibly Fake"
        else:
            status_str = "Genuine"
        result = {
            "result": {
                "fake_percentage": round(fake_percentage, 2),
                "status": status_str
            }
        }
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)

# -------------------------
# API Views
# -------------------------
class AudioDetectAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    @swagger_auto_schema(request_body=AudioUploadSerializer)
    def post(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return Response({"error": "No file provided."}, status=400)
        audio_file = request.FILES['file']
        model_path = os.path.join(os.path.dirname(__file__), "pretrained_models", "audio_classifier.h5")
        result = detect_fake_audio_from_file(audio_file, model_path)
        return Response(result)

class VideoDetectAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    @swagger_auto_schema(request_body=VideoUploadSerializer)
    def post(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return Response({"error": "No file provided."}, status=400)
        video_file = request.FILES['file']
        model_path = os.path.join(os.path.dirname(__file__), "pretrained_models", "model_84_acc_10_frames_final_data.pt")
        result = detect_fake_video_from_file(video_file, model_path)
        return Response(result)
