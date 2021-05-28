from django.db import models

# Create your models here.
class Post(models.Model):
    message=models.TextField()
    create=models.DateTimeField(auto_now=True)
    
class Video(models.Model):
    video_file=models.FileField(default='media/default.png',upload_to='CapD_rest/app/source')

class Person(models.Model):
    det_id=models.TextField()
    img_url=models.TextField()