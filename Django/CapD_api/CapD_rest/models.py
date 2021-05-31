from django.db import models

    
class Video(models.Model):
    video_file=models.FileField(default='media/default.png',upload_to='CapD_rest/app/source')

class Person(models.Model):
    det_id=models.TextField()
    img_url=models.TextField()

class Selected_Person(models.Model):
    selected_list=models.TextField(default='-1')

class Download(models.Model):
    url=models.TextField(default='-1')