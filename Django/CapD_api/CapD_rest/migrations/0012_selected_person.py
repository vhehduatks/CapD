# Generated by Django 3.0.14 on 2021-05-28 07:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('CapD_rest', '0011_auto_20210528_0301'),
    ]

    operations = [
        migrations.CreateModel(
            name='Selected_Person',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('selected_list', models.TextField()),
            ],
        ),
    ]
