# from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet
from .serializers import PostSerializer
from .models import Post

# Create your views here.
class PostViewSet(ModelViewSet):
    queryset =Post.objects.all()
    serializer_class=PostSerializer