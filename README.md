# Deepfake Detection System

## Overview
This project is an API-based deepfake detection system for both audio and video files. The system leverages deep learning models to classify audio and video inputs as genuine or fake. It provides two API endpoints:

1. **Audio Detection Endpoint** - Detects fake audio using a pre-trained deep learning model.
2. **Video Detection Endpoint** - Detects deepfake videos using a ResNeXt and LSTM-based model.

## Author

 - Developed by Prashanth

## Features
- REST API endpoints for uploading and analyzing audio and video files.
- Uses **librosa** for audio feature extraction and **TensorFlow/Keras** for classification.
- Uses **OpenCV**, **PyTorch**, and **ResNeXt** for video deepfake detection.
- JSON-based responses indicating the probability of the media being fake.
- Swagger documentation for easy API testing.

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Download Pretrained Models
The deepfake detection models need to be downloaded before running the API. You can download them from the following link:

[Download Models](https://drive.google.com/drive/folders/1Pnl0OESXnhsw4ys1epOUAKiFxR8KFmoV)

Place the downloaded models in the `pretrained_models/` directory within the project folder.

## Project Structure
```
├── deepfake_detection
│   ├── views.py           # API logic for audio and video deepfake detection
│   ├── serializers.py     # Serializers for API file uploads
│   ├── urls.py            # API routing
│   ├── pretrained_models  # Directory to store downloaded models
│   │   ├── audio_classifier.h5
│   │   ├── model_84_acc_10frames_final_data.pt
├── manage.py
├── requirements.txt       # Dependencies
├── README.md              # Documentation
```

## Usage
### Running the Django Server
Run the server using:
```bash
python manage.py runserver
```

### API Endpoints
#### 1. **Audio Detection API**
**Endpoint:** `/audio-detect/`

**Method:** `POST`

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/audio-detect/ \
    -F "file=@path_to_audio.wav"
```

**Response:**
```json
{
    "result": {
        "fake_percentage": 92.5,
        "status": "Highly Fake"
    }
}
```

#### 2. **Video Detection API**
**Endpoint:** `/video-detect/`

**Method:** `POST`

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/video-detect/ \
    -F "file=@path_to_video.mp4"
```

**Response:**
```json
{
    "result": {
        "fake_percentage": 78.3,
        "status": "Fake"
    }
}
```

## Model Details
### **Audio Model** (`audio_classifier.h5`)
- Deep learning model trained on spectrogram features extracted from audio.
- Uses **librosa** for feature extraction.
- Classifies the audio as **Genuine** or **Fake**.

### **Video Model** (`model_84_acc_10frames_final_data.pt`)
- **ResNeXt50** extracts spatial features from video frames.
- **LSTM** processes temporal dependencies.
- Analyzes 10 frames per video to classify them as **Genuine** or **Fake**.

## Technologies Used
- **Django Rest Framework (DRF)** - API framework for handling media uploads.
- **Librosa** - Audio feature extraction.
- **TensorFlow/Keras** - Audio deepfake detection model.
- **OpenCV** - Video processing.
- **PyTorch** - Video deepfake detection model.
- **Torchvision** - Image transformations.

## Contributing
1. Fork the repository.
2. Create a new feature branch (`feature-new-feature`).
3. Commit your changes.
4. Push to the branch.
5. Submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For questions or collaborations, feel free to reach out:
- Email: gummalaprashanth509@example.com
- GitHub: [YourGitHubProfile](https://github.com/Prashanth-TechAI)

---

This README provides all necessary details for setup, usage, and contribution. Let me know if you'd like any modifications!

