from django.conf.urls import url
from django.urls import path, include
from app import views as app_views

urlpatterns = [
    path('diagrams', app_views.diagram, name='diagrams'),
    path('show_modeler', app_views.show_modeler, name='show_modeler'),
    path('', app_views.index_app, name='index_app'),
]

