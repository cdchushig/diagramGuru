import os

from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from django.urls import reverse_lazy
from django.views import generic

from bootstrap_modal_forms.generic import BSModalCreateView, BSModalReadView

from app.models import Diagram
from .forms import DiagramModelForm

from xml.etree import ElementTree
from lxml import etree

from io import StringIO, BytesIO


PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(__file__))


class Index(generic.ListView):
    model = Diagram
    context_object_name = 'diagrams'
    template_name = 'list_diagrams.html'


class DiagramReadView(BSModalReadView):
    model = Diagram
    template_name = 'read_book.html'


class DiagramCreateView(BSModalCreateView):
    template_name = 'create_diagram.html'
    form_class = DiagramModelForm
    success_message = 'Success: Book was created.'
    success_url = reverse_lazy('list_bpmn')


def modeler(request):
    bpmn_filename = 'https://cdn.rawgit.com/bpmn-io/bpmn-js-examples/dfceecba/starter/diagram.bpmn'
    context = {'bpmn_filename': bpmn_filename}
    template = 'modeler.html'
    return render(request, template, context)


def list_bpmn(request):
    list_diagram = Diagram.objects.all()
    context = {"bpmn_list": list_diagram}
    template = 'list_diagrams.html'
    return render(request, template, context)


def create_new_diagram(request):
    bpmn_filename = os.path.join(settings.BASE_DIR, 'static', 'bpmn', 'diagrams', 'default.bpmn')
    with open(bpmn_filename, 'r') as f:
        bpmn_file_content = f.read()
    context = {'bpmn_filename': bpmn_filename, 'bpmn_file_content': bpmn_file_content, 'id_bpmn': -1}
    template = 'modeler.html'
    return render(request, template, context)


def save_bpmn(request):
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
        print(err)
        result_msg = err.message
        result_status = 0

    return HttpResponse(content_type="application/json", content='{"status":"%d", "msg":"%s"}' % (result_status, result_msg))


def open_bpmn(request, id):
    try:
        print('holi****')
        qs = Diagram.objects.filter(id=id)
        print(qs[0])
        print('holi****')
        if qs.exists():
            bpmn = qs[0]
            bpmn_file_content = bpmn.xml_content
            context = {'bpmn_file_content': bpmn_file_content, 'id_bpmn': bpmn.id}
            return render(request, 'bpmn/modeler.html', context)

    except Exception as err:
        print('exception')
        print(err)


def open_external_bpmn(request):

    output_directory = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'
    path_bpmn_file = output_directory + 'test3.xml'

    # parser = etree.XMLParser(ns_clean=True)
    tree = etree.parse(path_bpmn_file)
    # tree = etree.parse(StringIO(xml), parser)
    xml_str = etree.tostring(tree.getroot())

    # xml_tree = ET.parse(path_bpmn_file)
    # xml_str = ET.tostring(xml_tree, encoding='utf8', method='xml')

    # bpmn_file_content = bpmn.xml_content
    context = {'bpmn_file_content': xml_str, 'id_bpmn': 1}
    return render(request, 'modeler.html', context)


def delete_bpmn(request, id):
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
