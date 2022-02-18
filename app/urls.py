from django.contrib import admin
from django.urls import path, include
from app import views as app_views

urlpatterns = [
    path('admin', admin.site.urls),
    path('accounts', include("django.contrib.auth.urls")),
    path('login', app_views.login_request, name='login_request'),
    path('signup', app_views.signup, name='signup'),
    path('dashboard', app_views.show_dashboard, name='show_dashboard'),
    path('diagrams', app_views.diagram, name='diagrams'),
    path('show_model', app_views.show_model, name='show_model'),
    path('create_model', app_views.create_model, name='create_model'),
    # path('save_diagram', app_views.save_diagram, name='save_diagram'),
    path('save_model', app_views.save_model, name='save_model'),
    path('list_diagram', app_views.list_diagram, name='list_diagram'),
    # path('open_diagram/<int:id>', app_views.open_diagram, name="open_diagram"),
    path('open_external_diagram', app_views.open_external_diagram, name="open_external_diagram"),
    path('', app_views.index_app, name='index_app'),
]

