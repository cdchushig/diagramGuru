# Generated by Django 3.1.6 on 2021-04-20 15:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0004_auto_20210420_1548'),
    ]

    operations = [
        migrations.AlterField(
            model_name='diagram',
            name='author',
            field=models.CharField(default='', max_length=50),
        ),
        migrations.AlterField(
            model_name='diagram',
            name='comment',
            field=models.CharField(default='', max_length=300),
        ),
    ]
