from django.db import models


class Diagram(models.Model):
    name = models.CharField(max_length=255)
    author = models.CharField(max_length=50, default='')
    # date = models.DateField()
    # image = models.ImageField(null=True, blank=True)
    comment = models.CharField(max_length=300, default='')
    xml_content = models.TextField(max_length=0)




