# Generated by Django 3.2.4 on 2022-01-16 19:36

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('app', '0006_auto_20220116_1843'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='diagram',
            options={'ordering': ('-created',)},
        ),
        migrations.RemoveField(
            model_name='diagram',
            name='creation_date',
        ),
        migrations.AddField(
            model_name='diagram',
            name='created',
            field=models.DateField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='diagram',
            name='author',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='diagrams', to=settings.AUTH_USER_MODEL),
        ),
    ]
