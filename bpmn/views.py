import os

from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from django.urls import reverse_lazy
from django.views import generic
from django.template import Context, Template

from bootstrap_modal_forms.generic import BSModalCreateView, BSModalReadView

from app.models import Diagram
from .forms import DiagramModelForm

import logging
import coloredlogs
from lxml import etree
from xml.etree import ElementTree

from io import StringIO, BytesIO


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
    # bpmn_filename = 'https://cdn.rawgit.com/bpmn-io/bpmn-js-examples/dfceecba/starter/diagram.bpmn'
    # context = {'bpmn_filename': bpmn_filename}
    return render(request, template_name='modeler.html')


def list_diagram(request):
    logger.info('List diagrams')
    list_diagram = Diagram.objects.all()
    context = {"bpmn_list": list_diagram}
    template = 'list_diagram.html'
    return render(request, template, context)


def create_new_diagram(request):
    bpmn_filename = os.path.join(settings.BASE_DIR, 'static', 'bpmn', 'diagrams', 'default.bpmn')
    with open(bpmn_filename, 'r') as f:
        bpmn_file_content = f.read()
    context = {'bpmn_filename': bpmn_filename, 'bpmn_file_content': bpmn_file_content, 'id_bpmn': -1}
    template = 'modeler.html'
    return render(request, template, context)


def save_diagram(request):
    try:
        id = request.POST.get("id")
        name = request.POST.get("name")
        xml_content = request.POST.get("xml_content")

        if id and id != '-1': # if id was given, then update the diagram and don't create a new one
            qs = Diagram.objects.filter(id=id)
            if qs.exists():
                bpmn = qs[0]
                bpmn.xml_content = xml_content
                bpmn.save()
                result_msg = "BPMN updated sucessfully!"
                result_status = 2  # TODO: create an enum or choices to hold this status values
        else:
            # create a new diagram
            bpmn = Diagram.objects.create(name=name, xml_content=xml_content)
            bpmn.save()
            result_msg = "BPMN saved sucessfully!"
            result_status = 1  # TODO: create an enum or choices to hold this status values

    except Exception as err:
        logger.error(err)
        result_msg = err.message
        result_status = 0

    return HttpResponse(content_type="application/json", content='{"status":"%d", "msg":"%s"}' % (result_status, result_msg))


def open_diagram(request, id):
    try:
        qs = Diagram.objects.filter(id=id)
        if qs.exists():
            bpmn = qs[0]
            logger.info(bpmn)
            bpmn_file_content = bpmn.xml_content
            context = {'bpmn_file_content': bpmn_file_content, 'id_bpmn': bpmn.id}
            return render(request, 'bpmn/modeler.html', context)

    except Exception as err:
        logger.error('Exception')
        logger.error(err)


def open_external_diagram(request):
    bpmn_filename_xml = request.session.get('diagram_name')
    bpmn_path_filename_xml = PATH_DIAGRAM_GURU_DIAGRAMS + bpmn_filename_xml + '.xml'
    tree = etree.parse(bpmn_path_filename_xml)
    xml_str = etree.tostring(tree.getroot())
    context = {'bpmn_file_content': xml_str, 'id_bpmn': 1}

    logger.info('Loaded xml file: %s', request.session.get('diagram_name'))

    return render(request, 'modeler.html', context)


def delete_diagram(request, id):
    try:
        qs = Diagram.objects.filter(id=id)
        if qs.exists():
            name = qs[0].name
            msg = u"Diagram '%s' deleted successfully!" % name
            qs.delete()
        else:
            msg = None
            tag_msg = None
        tag_msg = 'success'
    except Exception as err:
        msg = err.message
        tag_msg = 'error'
    bpmn_list = Diagram.objects.all()
    context = {'bpmn_list':bpmn_list, 'msg': msg, 'tag_msg': tag_msg}
    template = 'bpmndesigner/list.html'
    return render(request, template, context)
