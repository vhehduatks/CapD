from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from .serializers import UploadSerializer
from .models import Video

import os
import sys

class UploadViewset(ModelViewSet):
    queryset=Video.objects.all()
    serializer_class = UploadSerializer
    def create(self, request):
            data=request.data
            model_video=UploadSerializer(data=data)
            if model_video.is_valid():
                model_video.save()
                return Response("upload file")
            else:
                return Response(status=404)

    
    def list(self,request):
        serializer = UploadSerializer(Video.objects.all(), many=True)
        return Response(serializer.data)


#post : 
class DetectorViewset(ModelViewSet):
    pass

class SelectIDViewset(ModelViewSet):
    pass