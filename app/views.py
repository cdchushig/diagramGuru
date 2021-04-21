import os
import sys

from django.http import HttpResponseRedirect
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Diagram
from .serializers import DiagramSerializer

# import bpmn_python.bpmn_diagram_layouter as layouter
# import bpmn_e2_python.bpmn_e2_diagram_rep as diagram

from bpmn_e2_python.bpmn_e2_diagram_rep import BpmnE2DiagramGraph
from bpmn_python.bpmn_diagram_layouter import generate_layout
from bpmn_python.bpmn_diagram_visualizer import bpmn_diagram_to_png

# from .detector.main import detect_diagram_objects

PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(__file__))


def index_app(request):
    return render(request, 'index2.html')


def show_dashboard(request):

    if request.method == 'POST':
        d = build_dictionary_with_automatic_analysis(request)
        return HttpResponseRedirect('/bpmn/open_external_bpmn')
    else:
        return HttpResponseRedirect('/bpmn/list_diagrams')


def build_dictionary_with_automatic_analysis(request):
    dict_metrics = {}

    if "_upload" in request.POST:
        dict_metrics, filename = _make_analysis_by_upload(request)
    elif '_url' in request.POST:
        # dict_metrics = _make_analysis_by_url(request)
        url = request.POST['urlProject']

    return dict_metrics


def _make_analysis_by_upload(request):
    try:
        diagram_file_uploaded = request.FILES['zipFile']
    except:
        d = {'Error': 'MultiValueDict'}
        return d

    dir_diagrams = PATH_DIAGRAM_GURU_PROJECT + "/uploads/"
    name_diagram_file_uploaded = diagram_file_uploaded.name
    path_diagram_file_uploaded = dir_diagrams + name_diagram_file_uploaded

    with open(path_diagram_file_uploaded, 'wb+') as destination:
        for chunk in diagram_file_uploaded.chunks():
            destination.write(chunk)

    dict_model = {}

    # stream = os.popen('ls -la')
    # output = stream.readlines()

    metricMastery = "python /home/cdchushig/multiverse/diagramGuru/detector/hello.py --file_name " + path_diagram_file_uploaded
    resultMastery = os.popen(metricMastery).read()
    list_nodes = resultMastery.split('\n')
    print(list_nodes)

    build_bpmn_diagram_from_nodes(list_nodes, name_diagram_file_uploaded)


    # detect_diagram_objects(path_diagram_file_uploaded)

    return dict_model, name_diagram_file_uploaded


def extract_diagram_type_and_position(str_node):
    if str_node != '':
        type_diagram_object = str_node.split('class:')[1].split(',')[0]
        return type_diagram_object
    else:
        return ''


def build_bpmn_diagram_from_nodes(list_nodes, name_diagram_file_uploaded):
    print('build_bpmn_diagram_from_nodes')

    output_directory = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'
    name_diagram_file_uploaded_without_ext = name_diagram_file_uploaded.split('.')[0]
    output_file = name_diagram_file_uploaded_without_ext + '.xml'

    bpmn_graph = BpmnE2DiagramGraph()
    bpmn_graph.create_new_diagram_graph(diagram_name="diagram1")
    process_id = bpmn_graph.add_process_to_diagram()

    [start_id, _] = bpmn_graph.add_start_event_to_diagram(process_id, start_event_name="")

    for diagram_node in list_nodes:
        print(diagram_node)
        type_diagram_object = extract_diagram_type_and_position(diagram_node)
        if type_diagram_object == 'process':
            [task1_id, _] = bpmn_graph.add_task_to_diagram(process_id, task_name="")
        elif type_diagram_object == 'decision':
            [task2_id, _] = bpmn_graph.add_parallel_gateway_to_diagram(process_id)

    # [task1_id, _] = bpmn_graph.add_task_to_diagram(process_id, task_name="")
    # [task2_id, _] = bpmn_graph.add_exclusive_gateway_to_diagram(process_id)
    # [task2_id, _] = bpmn_graph.add_parallel_gateway_to_diagram(process_id)

    [end_id, _] = bpmn_graph.add_end_event_to_diagram(process_id, end_event_name="")

    bpmn_graph.add_sequence_flow_to_diagram(process_id, start_id, task1_id,  "")

    # bpmn_graph.add_sequence_flow_to_diagram(process_id, task1_id, task2_id, "")
    # bpmn_graph.add_sequence_flow_to_diagram(process_id, task2_id, end_id, "")

    generate_layout(bpmn_graph)
    bpmn_graph.export_xml_file(output_directory, output_file)

    # bpmn_diagram_to_png(bpmn_graph, output_directory + name_diagram_file_uploaded_without_ext)


def load_bpmn_diagram(name_diagram_file_uploaded):
    output_directory = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'
    path_bpmn_file = output_directory + name_diagram_file_uploaded


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


