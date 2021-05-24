from rest_framework.serializers import ModelSerializer
from .models import Video


class UploadSerializer(ModelSerializer):
    class Meta:
        model=Video
        fields='__all__'
        