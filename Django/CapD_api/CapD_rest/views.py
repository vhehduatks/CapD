from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from .serializers import UploadSerializer
from .models import Video

class UploadViewset(ViewSet):
    serializer_class = UploadSerializer
    def create(self, request):
            data1=request.data
            model_video=UploadSerializer(data=data1)
            if model_video.is_valid():
                model_video.save()
                return Response("upload file")
            else:
                return Response(status=404)

    
    def list(self,request):
        serializer = UploadSerializer(Video.objects.all(), many=True)
        return Response(serializer.data)

    # @classmethod
    # def get_extra_actions(cls):
    #     return []