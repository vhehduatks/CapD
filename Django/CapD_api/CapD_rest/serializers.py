from django.db import models
from rest_framework.serializers import ModelSerializer,HyperlinkedModelSerializer,ImageField
from .models import Video,Person


class UploadSerializer(ModelSerializer):
    class Meta:
        model=Video
        fields='__all__'

class PersonSerializer(ModelSerializer):

    class Meta:
        model=Person
        fields='__all__'

# class PersonSerializer(HyperlinkedModelSerializer):
#     img=ImageField(use_url=True)
#     class Meta:
#         model=Person
#         fields=('det_id','img')