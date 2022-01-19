import datetime
from django.db import models
from django.contrib.auth.models import User, AbstractUser
from django.utils.translation import ugettext_lazy as _


class Diagram(models.Model):
    name = models.CharField(max_length=255)
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='diagrams')
    created = models.DateField(auto_now_add=True)
    image = models.ImageField(null=True, blank=True)
    comment = models.CharField(max_length=300, default='')
    xml_content = models.TextField(max_length=0)

    class Meta:
        ordering = ('-created',)




