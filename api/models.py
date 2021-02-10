from django.db import models
from django.contrib.auth.models import User


class BPMN(models.Model):
    xml_content = models.TextField(max_length=0)
    name = models.CharField(max_length=255)
    date = models.DateField()
    owner = models.ForeignKey(User)
