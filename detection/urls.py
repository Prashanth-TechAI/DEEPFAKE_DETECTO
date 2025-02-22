from django.urls import path
from .views import AudioDetectAPIView, VideoDetectAPIView

urlpatterns = [
    path('audio-detect/', AudioDetectAPIView.as_view(), name='audio-detect'),
    path('video-detect/', VideoDetectAPIView.as_view(), name='video-detect'),
]
