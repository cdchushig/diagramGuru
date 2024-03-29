# Generated by Django 3.2.4 on 2021-09-06 16:28

import datetime
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('bpmn', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='bpmn',
            name='date',
        ),
        migrations.AddField(
            model_name='bpmn',
            name='author',
            field=models.ForeignKey(default='', on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='bpmn',
            name='comment',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='bpmn',
            name='creation_date',
            field=models.DateTimeField(blank=True, default=datetime.datetime.now),
        ),
    ]
