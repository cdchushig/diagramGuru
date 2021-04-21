from django.db import models


class Bpmn(models.Model):
    xml_content = models.TextField(max_length=0)
    name = models.CharField(max_length=255)
    date = models.DateField()
    # owner = models.ForeignKey(User)