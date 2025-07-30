# chatbot_app/urls.py
from django.urls import path
from .views import GenerateAPIView, RecommendAPIView

urlpatterns = [
    path('generate', GenerateAPIView.as_view(), name='generate-api'),
    path('recommend', RecommendAPIView.as_view(), name='recommend-api'),
]