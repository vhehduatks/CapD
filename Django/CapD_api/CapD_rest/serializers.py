from django.db import models
from rest_framework.serializers import ModelSerializer,HyperlinkedModelSerializer,ImageField
from .models import Video,Person,Selected_Person,Download


class UploadSerializer(ModelSerializer):
    class Meta:
        model=Video
        fields='__all__'

class PersonSerializer(ModelSerializer):

    class Meta:
        model=Person
        fields='__all__'

class Selected_PersonSerializer(ModelSerializer):

    class Meta:
        model=Selected_Person
        fields='__all__'

class DownloadSerializer(ModelSerializer):
    class Meta:
        model=Download
        fields='__all__'