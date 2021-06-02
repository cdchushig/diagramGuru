import os
import logging
from datetime import datetime
import base64
import json
import requests
import subprocess

from django.http import HttpResponseRedirect
from django.shortcuts import render

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .forms import UploadFileForm
from .models import Diagram
from .serializers import DiagramSerializer
from .app_consts import Consts

from bpmn_python.bpmn_python.bpmn_diagram_layouter import generate_layout
from bpmn_python.bpmn_e2_python.bpmn_e2_diagram_rep import BpmnE2DiagramGraph
from bpmn_python.bpmn_python.bpmn_diagram_visualizer import bpmn_diagram_to_png


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(__file__))
PATH_DIR_UPLOADS = PATH_DIAGRAM_GURU_PROJECT + "/uploads/"
PATH_DIR_DIAGRAMS = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'


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


def do_cmd_to_shell_diagram_detector(diagram_path_filename):

    script_path_filename = PATH_DIAGRAM_GURU_PROJECT + '/diagram_detector/detector_main.py'

    cmd_diagram_detector = ["python", script_path_filename, "--diagram_filename", diagram_path_filename]
    process = subprocess.Popen(cmd_diagram_detector, stdout=subprocess.PIPE, stderr=None)
    cmd_output = process.communicate()

    dict_objects_str = cmd_output[0].decode("utf-8")
    dict_objects_str = str(dict_objects_str).replace("'", '"')
    dict_objects = eval(dict_objects_str)

    return dict_objects


def do_request_to_api_diagram_detector(img_bytes, diagram_filename_uploaded_unique_id_with_extension):
    im_b64 = base64.b64encode(img_bytes).decode("utf8")
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = json.dumps({"image": im_b64, "diagram_name": diagram_filename_uploaded_unique_id_with_extension})
    response = requests.post(Consts.api_url, data=payload, headers=headers)

    try:
        dict_objects = response.json()
    except requests.exceptions.RequestException:
        print(response.text)

    return dict_objects


def save_file_uploaded(diagram_file_uploaded, diagram_path_filename_uploaded):
    with open(diagram_path_filename_uploaded, 'wb+') as destination:
        for chunk in diagram_file_uploaded.chunks():
            destination.write(chunk)


def get_filename_unique_id_and_ext(diagram_filename_with_extension):
    unique_id = datetime.now().strftime('_%Y_%m_%d_%H%M%S')
    list_diagram_filename = diagram_filename_with_extension.split('.')
    ext_file = '.' + list_diagram_filename[1]
    diagram_filename_unique_id = list_diagram_filename[0] + unique_id
    return diagram_filename_unique_id, ext_file


def handle_uploaded_file(diagram_file_uploaded):
    diagram_filename_with_extension = diagram_file_uploaded.name
    diagram_filename_unique_id, ext_file = get_filename_unique_id_and_ext(diagram_filename_with_extension)
    diagram_path_filename_unique_id = PATH_DIR_UPLOADS + diagram_filename_unique_id + ext_file
    save_file_uploaded(diagram_file_uploaded, diagram_path_filename_unique_id)

    dict_diagram_objects = do_cmd_to_shell_diagram_detector(diagram_path_filename_unique_id)

    build_bpmn_from_nodes(dict_diagram_objects, diagram_filename_unique_id)

    return diagram_filename_unique_id


def extract_diagram_type_and_position(dict_node):
    type_diagram_object = dict_node['class_shape']
    vector_positions = dict_node['coordinate']
    return type_diagram_object, vector_positions


def build_bpmn_from_nodes(dict_objects, diagram_filename_unique_id):
    """
    Build a BPMN file with a dictionary of elements linked to a diagram
    :param dict_objects: Dictionary with elements of diagram
    :param diagram_filename_unique_id:
    :return:
    """
    logging.info("Build bpmn from nodes")

    output_bpmn_file = diagram_filename_unique_id + '.xml'

    bpmn_graph = BpmnE2DiagramGraph()
    bpmn_graph.create_new_diagram_graph(diagram_name="diagram1")
    process_id = bpmn_graph.add_process_to_diagram()

    # [start_id, _] = bpmn_graph.add_start_event_to_diagram(process_id, start_event_name="")

    for dict_diagram_node in dict_objects['nodes']:
        type_diagram_object, vector_positions = extract_diagram_type_and_position(dict_diagram_node)

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
    bpmn_graph.export_xml_file(PATH_DIR_DIAGRAMS, output_bpmn_file)

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



