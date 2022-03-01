import os
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from django.urls import reverse_lazy
from django.views import generic

from bootstrap_modal_forms.generic import BSModalCreateView, BSModalReadView

from app.models import Diagram
from .forms import DiagramModelForm

import logging
import coloredlogs
from lxml import etree

PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(__file__))
PATH_DIAGRAM_GURU_DIAGRAMS = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


class Index(generic.ListView):
    model = Diagram
    context_object_name = 'diagrams'
    template_name = 'list_diagram.html'


class DiagramReadView(BSModalReadView):
    model = Diagram
    template_name = 'read_book.html'


class DiagramCreateView(BSModalCreateView):
    template_name = 'create_diagram.html'
    form_class = DiagramModelForm
    success_message = 'Success: Book was created.'
    success_url = reverse_lazy('list_diagram')


def modeler(request):
    return render(request, template_name='modeler_oc.html')


# def list_diagram(request):
#     logger.info('List diagrams')
#     list_diagram = Diagram.objects.all()
#     context = {"bpmn_list": list_diagram}
#     template = 'list_diagram.html'
#     return render(request, template, context)


# def create_new_diagram(request):
#     bpmn_filename = os.path.join(settings.BASE_DIR, 'static', 'bpmn', 'diagrams', 'default.bpmn')
#     with open(bpmn_filename, 'r') as f:
#         bpmn_file_content = f.read()
#     context = {'bpmn_filename': bpmn_filename, 'bpmn_file_content': bpmn_file_content, 'id_bpmn': -1}
#     template = 'modeler.html'
#     return render(request, template, context)






