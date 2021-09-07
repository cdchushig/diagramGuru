from django.db import models
from django.contrib.auth.models import User
from datetime import datetime


class Bpmn(models.Model):
    xml_content = models.TextField(max_length=0)
    name = models.CharField(max_length=255)
    author = models.ForeignKey(User, default="", on_delete=models.CASCADE)
    comment = models.TextField(default="")
    creation_date = models.DateTimeField(default=datetime.now, blank=True)
