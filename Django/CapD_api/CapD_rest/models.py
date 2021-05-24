from django.db import models

# Create your models here.
class Post(models.Model):
    message=models.TextField()
    create=models.DateTimeField(auto_now=True)
    
class Video(models.Model):
    title=models.CharField(default='test1',max_length=200)
    video_file=models.FileField(default='media/default.png',upload_to='upload')