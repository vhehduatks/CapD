# Generated by Django 3.0.14 on 2021-05-24 06:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('CapD_rest', '0004_auto_20210524_0545'),
    ]

    operations = [
        migrations.AlterField(
            model_name='video',
            name='video_file',
            field=models.FileField(default='media/default.png', upload_to='upload'),
        ),
    ]