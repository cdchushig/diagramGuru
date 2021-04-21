from django.urls import path

from bpmn import views as bpmn_views

urlpatterns = [
    path('list_diagrams', bpmn_views.list_bpmn, name='list_bpmn'),
    path('save_bpmn', bpmn_views.save_bpmn, name='save_bpmn'),
    path('create_new_diagram', bpmn_views.create_new_diagram, name="create_new_diagram"),
    path('create_bpmn', bpmn_views.DiagramCreateView.as_view(), name="create_bpmn"),
    path('read_bpmn/<int:pk>', bpmn_views.DiagramReadView.as_view(), name='read_bpmn'),
    path('open_bpmn/<int:id>', bpmn_views.open_bpmn, name="open_bpmn"),
    path('open_external_bpmn', bpmn_views.open_external_bpmn, name="open_external_bpmn"),
    path('delete_bpmn/(\d+)/', bpmn_views.delete_bpmn, name="delete_bpmn"),
    path('modeler/', bpmn_views.modeler, name='modeler'),
    path('', bpmn_views.modeler, name='modeler'),
]

# path('read/<int:pk>', views.BookReadView.as_view(), name='read_book'),
# path('<int:question_id>/', views.detail, name='detail'),

