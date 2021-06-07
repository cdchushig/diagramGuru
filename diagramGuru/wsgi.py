"""
WSGI config for diagramGuru project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'diagramGuru.settings')
os.environ['DJANGO_SETTINGS_MODULE'] = 'diagramGuru.settings'

# application = get_wsgi_application()

import django
django.setup()

import django.core.handlers.wsgi
application = django.core.handlers.wsgi.WSGIHandler()

