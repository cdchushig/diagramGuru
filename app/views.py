import os

from datetime import datetime
import base64
import json
import requests
import subprocess
import itertools as it
import operator


from django.http import HttpResponseRedirect
from django.shortcuts import render

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .forms import UploadFileForm
from .models import Diagram
from .serializers import DiagramSerializer
from .app_consts import Consts

import networkx as nx

from .diagram_node import DiagramNode
from app.diagram_utils import compute_distance_between_nodes

from bpmn_python.bpmn_python.bpmn_diagram_layouter import generate_layout
from bpmn_python.bpmn_e2_python.bpmn_e2_diagram_rep import BpmnE2DiagramGraph

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

DIST_MIN_EDGING = 420
PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(__file__))
PATH_DIR_UPLOADS = PATH_DIAGRAM_GURU_PROJECT + "/uploads/"
PATH_DIR_DIAGRAMS = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'

list_diagram_types_allowed = ['process', 'decision', 'start_end', 'scan']


def index_app(request):
    return render(request, 'index2.html')


def show_dashboard(request):
    """
    Handle Request POST with uploded file
    :param request: Request object.
    :return:
    """
    if request.method == 'POST':
        upload_file_form = UploadFileForm(request.POST, request.FILES)
        if upload_file_form.is_valid():
            diagram_name_unique_id = handle_uploaded_file(request.FILES['diagram_file'])
            request.session['diagram_name'] = diagram_name_unique_id
            return HttpResponseRedirect('/bpmn/open_external_bpmn')
    else:
        return HttpResponseRedirect('/')


def do_cmd_to_shell_diagram_detector(diagram_path_filename):
    """
    Execute a cmd command and return output string
    :param diagram_path_filename: String object.
    :return: dict_objects. Dictionary object. Dictionary with diagram objects.
    """
    logging.info("do_cmd_to_shell_diagram_detector")

    script_path_filename = PATH_DIAGRAM_GURU_PROJECT + '/diagram_detector/detector_main.py'

    cmd_diagram_detector = ["python", script_path_filename, "--diagram_filename", diagram_path_filename]
    process = subprocess.Popen(cmd_diagram_detector, stdout=subprocess.PIPE, stderr=None)
    cmd_output = process.communicate()

    dict_objects_str = cmd_output[0].decode("utf-8")
    dict_objects_str = str(dict_objects_str).replace("'", '"')
    dict_objects = eval(dict_objects_str)

    return dict_objects


def do_request_to_api_diagram_detector(img_bytes, diagram_filename_uploaded_unique_id_with_extension):
    """
    Do request to API diagram detector.
    :param img_bytes:
    :param diagram_filename_uploaded_unique_id_with_extension:
    :return:
    """
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
    """
    Save diagram file upload to local folder.
    :param diagram_file_uploaded:
    :param diagram_path_filename_uploaded:
    :return:
    """
    with open(diagram_path_filename_uploaded, 'wb+') as destination:
        for chunk in diagram_file_uploaded.chunks():
            destination.write(chunk)


def get_filename_unique_id_and_ext(diagram_filename_with_extension):
    """
    Get a filename with an unique id
    :param diagram_filename_with_extension: String object. Name of file uploaded with extension.
    :return: String object.
    """
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
    diagram_graph = create_graph_from_list_nodes(dict_diagram_objects, verbose=1)
    transform_graph_to_bpmn(diagram_graph, diagram_filename_unique_id)

    return diagram_filename_unique_id


def create_bpmn_graph_element(diagram_node, process_id, bpmn_graph):
    # list_diagram_types_allowed = ['process', 'decision', 'start_end', 'scan']
    type_diagram_element = diagram_node.get_type()
    print(type_diagram_element)
    if type_diagram_element == 'start_end':
        return bpmn_graph.add_task_to_diagram(process_id, task_name="")
    elif type_diagram_element == 'scan':
        return bpmn_graph.add_task_to_diagram(process_id, task_name="")
    elif type_diagram_element == 'process':
        return bpmn_graph.add_task_to_diagram(process_id, task_name="")
    elif type_diagram_element == 'decision':
        return bpmn_graph.add_parallel_gateway_to_diagram(process_id, "")


def create_graph_from_list_nodes(dict_objects, verbose=0):
    """
    Create a DiagramGraph object with diagram objects (nodes)
    :param verbose:
    :param dict_objects:
    :return:
    """
    logging.info("create_graph_from_list_nodes")

    list_diagram_nodes = []
    diagram_graph = nx.Graph()
    for count, node in enumerate(dict_objects['nodes']):
        diagram_node = DiagramNode(count, node['coordinate'], node['class_shape'], node['text'])
        list_diagram_nodes.append(diagram_node)
        if verbose:
            print(diagram_node)

    list_diagram_nodes_ordered = order_nodes_by_center_positions(list_diagram_nodes)

    for node_item in list_diagram_nodes_ordered:
        diagram_graph.add_node(node_item.id, data=node_item)

    dict_connected_nodes = build_dictionary_with_connected_nodes(diagram_graph)
    list_tuples_connected_nodes = build_list_tuples_connected_nodes(dict_connected_nodes, diagram_graph)
    diagram_graph = update_diagram_graph_with_edges(diagram_graph, list_tuples_connected_nodes)

    return diagram_graph


def update_diagram_graph_with_edges(diagram_graph, list_tuples_connected_nodes):
    for tuple_node in list_tuples_connected_nodes:
        diagram_graph.add_edge(tuple_node[0], tuple_node[1])

    return diagram_graph


def build_dictionary_with_connected_nodes(diagram_graph):
    """
    Build a Dictionary with ids of connected nodes
    :param diagram_graph:
    :return:
    """
    dict_tuples_nodes = {}
    for n1, n2 in it.combinations(diagram_graph.nodes(), 2):
        if n1 in dict_tuples_nodes.keys():
            list_id_nodes = dict_tuples_nodes.get(n1)
            list_id_nodes.append(n2)
            dict_tuples_nodes.update({n1: list_id_nodes})
        else:
            dict_tuples_nodes[n1] = [n2]

    return dict_tuples_nodes


def build_list_tuples_connected_nodes(dict_tuples_nodes, diagram_graph):
    """
    Build list of tuples of nodes which are connected according to distance
    :param dict_tuples_nodes:
    :param diagram_graph:
    :return:
    """
    list_node_edges = []
    for id_node_x, list_nodes in dict_tuples_nodes.items():
        list_aux_tuples = []
        for id_node_y in list_nodes:
            dist_nodes = compute_distance_between_nodes(diagram_graph.node[id_node_x]['data'],
                                                        diagram_graph.node[id_node_y]['data'])
            tuple_node = (id_node_x, id_node_y, dist_nodes)
            list_aux_tuples.append(tuple_node)

        list_node_edges.extend(get_nodes_with_minimum_distance(list_aux_tuples))

    return list_node_edges


def get_nodes_with_minimum_distance(list_nodes):
    list_nodes_connected = []
    min_node = min(list_nodes, key=operator.itemgetter(2))

    for tuple_node in list_nodes:
        if min_node[2] <= tuple_node[2] <= DIST_MIN_EDGING:
            list_nodes_connected.append(tuple_node)

    return list_nodes_connected


def order_nodes_by_center_positions(list_nodes):
    """
    Order list of DiagramNode by center positions
    :param list_nodes: List of DiagramNode.
    :return:
    """
    list_nodes_sorted = sorted(list_nodes, key=lambda x: [x.centers[1], x.centers[0]])
    list_nodes_sorted_filtered = list(filter(lambda x: x.type in list_diagram_types_allowed, list_nodes_sorted))
    return list_nodes_sorted_filtered


def transform_graph_to_bpmn(diagram_graph, diagram_filename_unique_id):
    """
    Build a BPMN file with a dictionary of elements linked to a diagram
    :param diagram_graph: DiagramGraph object.
    :param diagram_filename_unique_id: string object.
    :return:
    """
    logging.info("transform_graph_to_bpmn")

    bpmn_graph = BpmnE2DiagramGraph()
    bpmn_graph.create_new_diagram_graph(diagram_name="diagram1")
    process_id = bpmn_graph.add_process_to_diagram()

    print('diagram_graph')
    print(*diagram_graph)
    print('----')

    task_id_src = 1000
    task_id_dst = 1000
    for tuple_connected_nodes in diagram_graph.edges:

        [task1_id, _] = create_bpmn_graph_element(diagram_graph.node[tuple_connected_nodes[0]]['data'],
                                                  process_id,
                                                  bpmn_graph)

        [task2_id, _] = create_bpmn_graph_element(diagram_graph.node[tuple_connected_nodes[1]]['data'],
                                                  process_id,
                                                  bpmn_graph)

        bpmn_graph.add_sequence_flow_to_diagram(process_id, task1_id, task2_id, "")
        task_id_src = task2_id

    generate_layout(bpmn_graph)
    bpmn_graph.export_xml_file(PATH_DIR_DIAGRAMS, diagram_filename_unique_id + '.xml')


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



