import os
import sys

from datetime import datetime
from uuid import uuid4

from django.template import Context, Template

from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.urls import reverse

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .forms import UploadFileForm
from .models import Diagram
from .serializers import DiagramSerializer

# import bpmn_python.bpmn_diagram_layouter as layouter
# import bpmn_e2_python.bpmn_e2_diagram_rep as diagram

from bpmn_e2_python.bpmn_e2_diagram_rep import BpmnE2DiagramGraph
from bpmn_python.bpmn_diagram_layouter import generate_layout
from bpmn_python.bpmn_diagram_visualizer import bpmn_diagram_to_png
import base64
import json

import requests

# from .detector.main import detect_diagram_objects

PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(__file__))
PATH_DIR_DIAGRAMS = PATH_DIAGRAM_GURU_PROJECT + "/uploads/"
PATH_DIR_BPMN = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'
api_url = 'http://127.0.0.1:5000/api/v1/detect'


def index_app(request):
    return render(request, 'index2.html')


def show_dashboard(request):
    if request.method == 'POST':
        upload_file_form = UploadFileForm(request.POST, request.FILES)
        if upload_file_form.is_valid():
            diagram_name_unique_id = handle_uploaded_file(request.FILES['diagram_file'])
            request.session['diagram_name'] = diagram_name_unique_id
            return HttpResponseRedirect('/bpmn/open_external_bpmn')
    else:
        return HttpResponseRedirect('/')


def handle_uploaded_file(diagram_file_uploaded):

    unique_id = datetime.now().strftime('_%Y_%m_%d_%H%M%S')

    name_diagram_file_uploaded_with_extension = diagram_file_uploaded.name
    ext_file = '.' + name_diagram_file_uploaded_with_extension.split('.')[1]
    name_diagram_file_uploaded_unique_id = name_diagram_file_uploaded_with_extension.split('.')[0] + unique_id
    path_diagram_file_uploaded = PATH_DIR_DIAGRAMS + name_diagram_file_uploaded_unique_id + ext_file
    name_diagram_file_uploaded_unique_id_with_extension = name_diagram_file_uploaded_unique_id + ext_file

    with open(path_diagram_file_uploaded, 'wb+') as destination:
        for chunk in diagram_file_uploaded.chunks():
            destination.write(chunk)

    with open(path_diagram_file_uploaded, "rb") as f:
        im_bytes = f.read()

    im_b64 = base64.b64encode(im_bytes).decode("utf8")
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = json.dumps({"image": im_b64, "diagram_name": name_diagram_file_uploaded_unique_id_with_extension})
    response = requests.post(api_url, data=payload, headers=headers)

    try:
        dict_objects = response.json()
    except requests.exceptions.RequestException:
        print(response.text)

    print(dict_objects)
    build_bpmn_from_nodes(dict_objects, name_diagram_file_uploaded_unique_id)

    return name_diagram_file_uploaded_unique_id


def extract_diagram_type_and_position(dict_node):
    type_diagram_object = dict_node['class_shape']
    vector_positions = dict_node['coordinate']
    return type_diagram_object, vector_positions


def build_bpmn_from_nodes(dict_objects, diagram_name_unique):
    print('build_bpmn_diagram_from_nodes')

    output_bpmn_file = diagram_name_unique + '.xml'

    bpmn_graph = BpmnE2DiagramGraph()
    bpmn_graph.create_new_diagram_graph(diagram_name="diagram1")
    process_id = bpmn_graph.add_process_to_diagram()

    # [start_id, _] = bpmn_graph.add_start_event_to_diagram(process_id, start_event_name="")

    for dict_diagram_node in dict_objects['nodes']:
        type_diagram_object, vector_positions = extract_diagram_type_and_position(dict_diagram_node)

        print(type_diagram_object)

        if type_diagram_object == 'process':
            [task1_id, _] = bpmn_graph.add_task_to_diagram(process_id, task_name="")
        elif type_diagram_object == 'decision':
            [task2_id, _] = bpmn_graph.add_parallel_gateway_to_diagram(process_id)

    # [task1_id, _] = bpmn_graph.add_task_to_diagram(process_id, task_name="")
    # [task2_id, _] = bpmn_graph.add_exclusive_gateway_to_diagram(process_id)
    # [task2_id, _] = bpmn_graph.add_parallel_gateway_to_diagram(process_id)

    # [end_id, _] = bpmn_graph.add_end_event_to_diagram(process_id, end_event_name="")

    # bpmn_graph.add_sequence_flow_to_diagram(process_id, start_id, task1_id,  "")

    # bpmn_graph.add_sequence_flow_to_diagram(process_id, task1_id, task2_id, "")
    # bpmn_graph.add_sequence_flow_to_diagram(process_id, task2_id, end_id, "")

    generate_layout(bpmn_graph)
    bpmn_graph.export_xml_file(PATH_DIR_BPMN, output_bpmn_file)

    # bpmn_diagram_to_png(bpmn_graph, output_directory + name_diagram_file_uploaded_without_ext)


@api_view(['GET', 'POST'])
def diagram(request):
    if request.method == 'GET':
        diagrams = Diagram.objects.all()
        serializer = DiagramSerializer(diagrams, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        serializer = DiagramSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




# def build_dictionary_with_automatic_analysis(request):
#     dict_metrics = {}
#
#     if "_upload" in request.POST:
#         dict_metrics, filename = _make_analysis_by_upload(request)
#     elif '_url' in request.POST:
        dict_metrics = _make_analysis_by_url(request)
        # url = request.POST['urlProject']
    #
    # return dict_metrics, filename


# def _make_analysis_by_upload(request):
#     try:
#         diagram_file_uploaded = request.FILES['zipFile']
#     except:
#         d = {'Error': 'MultiValueDict'}
#         return d
#
#     dir_diagrams = PATH_DIAGRAM_GURU_PROJECT + "/uploads/"
#     name_diagram_file_uploaded = diagram_file_uploaded.name
#     path_diagram_file_uploaded = dir_diagrams + name_diagram_file_uploaded
#
#     with open(path_diagram_file_uploaded, 'wb+') as destination:
#         for chunk in diagram_file_uploaded.chunks():
#             destination.write(chunk)
#
#     with open(path_diagram_file_uploaded, "rb") as f:
#         im_bytes = f.read()
#
#     im_b64 = base64.b64encode(im_bytes).decode("utf8")
#     headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
#     payload = json.dumps({"image": im_b64, "other_key": name_diagram_file_uploaded})
    # response = requests.post(api_url, data=payload, headers=headers)

    # try:
    #     data = response.json()
    #     print(data)
    # except requests.exceptions.RequestException:
    #     print(response.text)

    # dict_model = {}
    # metricMastery = "python /home/cdchushig/multiverse/diagramGuru/detector/hello.py --file_name " + path_diagram_file_uploaded
    # resultMastery = os.popen(metricMastery).read()
    # list_nodes = resultMastery.split('\n')
    # build_bpmn_from_nodes(list_nodes, name_diagram_file_uploaded)
    # detect_diagram_objects(path_diagram_file_uploaded)
    # return dict_model, name_diagram_file_uploaded

# def analyze_diagram_local():
#     dict_model = {}
#     metricMastery = "python /home/cdchushig/multiverse/diagramGuru/detector/hello.py --file_name " + path_diagram_file_uploaded
#     resultMastery = os.popen(metricMastery).read()
#     list_nodes = resultMastery.split('\n')
#     print(list_nodes)

# def load_bpmn_diagram(name_diagram_file_uploaded):
#     output_directory = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'
#     path_bpmn_file = output_directory + name_diagram_file_uploaded
