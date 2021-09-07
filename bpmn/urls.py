from django.urls import path

from bpmn import views as bpmn_views

urlpatterns = [
    path('list_diagram', bpmn_views.list_diagram, name='list_diagram'),
    path('save_diagram', bpmn_views.save_diagram, name='save_diagram'),
    path('create_new_diagram', bpmn_views.create_new_diagram, name="create_new_diagram"),
    path('create_diagram', bpmn_views.DiagramCreateView.as_view(), name="create_diagram"),
    path('read_diagram/<int:pk>', bpmn_views.DiagramReadView.as_view(), name='read_diagram'),
    path('open_diagram/<int:id>', bpmn_views.open_diagram, name="open_diagram"),
    path('open_external_diagram', bpmn_views.open_external_diagram, name="open_external_diagram"),
    path('delete_diagram/(\d+)/', bpmn_views.delete_diagram, name="delete_diagram"),
    path('modeler/', bpmn_views.modeler, name='modeler'),
    path('', bpmn_views.modeler, name='modeler'),
]

