# Generated by Django 3.0.14 on 2021-05-25 07:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('CapD_rest', '0007_remove_video_title'),
    ]

    operations = [
        migrations.CreateModel(
            name='Person',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('det_id', models.TextField()),
                ('img', models.ImageField(upload_to='')),
            ],
        ),
        migrations.AlterField(
            model_name='video',
            name='video_file',
            field=models.FileField(default='media/default.png', upload_to='app/source'),
        ),
    ]
