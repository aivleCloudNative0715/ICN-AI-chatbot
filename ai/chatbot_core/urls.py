# chatbot_core/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('chatbot/', include('chatbot_app.urls')), # '/chatbot/message/' 로 접근 가능
]