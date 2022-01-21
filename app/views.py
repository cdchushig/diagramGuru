import os

from datetime import datetime
import base64
import json
import requests
import subprocess
import itertools as it
import operator

import random
import string

from django.http import HttpResponseRedirect
from django.contrib.auth import authenticate, login
from django.conf import settings
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib import messages

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .forms import UploadFileForm, NewUserForm, NewModelForm
from .models import Diagram
from .serializers import DiagramSerializer
from .app_consts import Consts

from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes

import networkx as nx
from xml.etree.ElementTree import tostring
import cv2

from .diagram_node import DiagramNode
from app.diagram_utils import compute_distance_between_nodes

from bpmn_python_lib2.bpmn_python.bpmn_diagram_export import BpmnDiagramGraphExport
from bpmn_python_lib2.bpmn_python.bpmn_diagram_layouter import generate_layout
from bpmn_python_lib2.bpmn_e2_python.bpmn_e2_diagram_rep import BpmnE2DiagramGraph

from lxml import etree

DIST_MIN_EDGING = 350
PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(__file__))
PATH_DIR_UPLOADS = PATH_DIAGRAM_GURU_PROJECT + "/uploads/"
PATH_DIR_DIAGRAMS = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'
PATH_VENV_DIAGRAM_GURU = ''

BPMN_MODEL_NS = 'http://www.omg.org/spec/BPMN/20100524/MODEL'
DIAG_INTERCHANGE_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
DIAG_COMMON_NS = "http://www.omg.org/spec/DD/20100524/DC"

list_diagram_types_allowed = ['process', 'decision', 'start_end', 'scan']

import logging
import coloredlogs
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def index_app(request):
    # username = request.POST['username']
    # password = request.POST['password']
    # user = authenticate(request, username=username, password=password)
    # if user is not None:
    #     login(request, user)
    #     return render(request, 'index2.html')
    # else:
    #     return render(request, 'index2.html')
    return render(request, 'index2.html')


def login_request(request):

    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}.")
                return redirect("/list_diagram")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")

    form = AuthenticationForm()

    return render(request=request, template_name="login.html", context={"login_form": form})


def signup(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful." )
            return redirect("/list_diagram")

        messages.error(request, "Unsuccessful registration. Invalid information.")

    form = NewUserForm()

    return render(request=request, template_name="signup.html", context={"register_form": form})


def save_model(request):

    if request.method == "POST":
        form = NewModelForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful." )
            return redirect("/list_diagram")

        messages.error(request, "Unsuccessful registration. Invalid information.")

    form = NewUserForm()

    return render(request=request, template_name="save_model.html", context={"register_form": form})


def show_dashboard(request):
    return render(request, "dashboard.html", context={})


def create_model(request):
    bpmn_filename = os.path.join(settings.BASE_DIR, 'static', 'bpmn', 'diagrams', 'default.bpmn')
    with open(bpmn_filename, 'r') as f:
        bpmn_file_content = f.read()
    context = {'bpmn_filename': bpmn_filename, 'bpmn_file_content': bpmn_file_content, 'id_bpmn': -1}
    template = 'modeler/modeler_oc.html'
    return render(request, template, context)


def list_diagram(request):
    logger.info('List diagrams')
    list_diagram = Diagram.objects.all()
    return render(request, 'list_diagram.html', {"bpmn_list": list_diagram})


def show_model(request):
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
            return HttpResponseRedirect('/bpmn/open_external_diagram')
    else:
        return HttpResponseRedirect('/')


def handle_404(request):
    data = {}
    return render(request, 'error.html', data)


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
        # logger.info(dict_objects)
    except requests.exceptions.RequestException:
        logger.error('Error in response from api')
        logger.error(response.text)

    return dict_objects


def convert_upload_format_file_to_png(input_path, output_path, ext_file):

    if ext_file in 'pdf':
        img_file = convert_from_path(input_path + 'example.pdf')
    elif ext_file in 'jpg':
        img_file = Image.open(input_path + 'PhotoName' + ext_file)
        img_file.save(output_path + 'PhotoName.png' + '.png')


def save_file_uploaded(diagram_file_uploaded, diagram_path_filename_uploaded):
    """
    Save diagram file to local folder.
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
    diagram_filename_unique_id, ext_file = get_filename_unique_id_and_ext(diagram_file_uploaded.name)
    diagram_path_filename_unique_id = PATH_DIR_UPLOADS + diagram_filename_unique_id + ext_file
    save_file_uploaded(diagram_file_uploaded, diagram_path_filename_unique_id)

    dict_diagram_objects = do_cmd_to_shell_diagram_detector(diagram_path_filename_unique_id)

    # diagram_img = cv2.imread(diagram_path_filename_unique_id)
    # is_success, im_buf_arr = cv2.imencode(ext_file, diagram_img)
    # img_bytes = im_buf_arr.tobytes()
    # dict_diagram_objects = do_request_to_api_diagram_detector(img_bytes, diagram_path_filename_unique_id)

    if dict_diagram_objects:
        diagram_graph, list_diagram_nodes_ordered = create_graph_from_list_nodes(dict_diagram_objects, verbose=1)
        transform_graph_to_bpmn(diagram_graph, list_diagram_nodes_ordered, diagram_filename_unique_id)
    else:
        logger.error('No objects in diagram recognized!')

    return diagram_filename_unique_id


def create_bpmn_graph_element(diagram_node, process_id, bpmn_graph):
    # list_diagram_types_allowed = ['process', 'decision', 'start_end', 'scan']
    type_diagram_element = diagram_node.get_type()
    logger.info('Added diagram object: %s', type_diagram_element)
    # text_element = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
    text_element = ''

    if type_diagram_element == 'start_end':
        return bpmn_graph.add_start_event_to_diagram(process_id, start_event_name=text_element)
    elif type_diagram_element == 'scan':
        return bpmn_graph.add_task_to_diagram(process_id,
                                              pos_x=diagram_node.compute_centers()[0],
                                              pos_y=diagram_node.compute_centers()[1],
                                              task_name=text_element)
    elif type_diagram_element == 'process':
        return bpmn_graph.add_task_to_diagram(process_id,
                                              pos_x=diagram_node.compute_centers()[0],
                                              pos_y=diagram_node.compute_centers()[1],
                                              task_name=text_element)
    elif type_diagram_element == 'decision':
        return bpmn_graph.add_parallel_gateway_to_diagram(process_id,
                                                          pos_x=diagram_node.compute_centers()[0],
                                                          pos_y=diagram_node.compute_centers()[1],
                                                          gateway_name='')
    else:
        return bpmn_graph.add_task_to_diagram(process_id,
                                              pos_x=diagram_node.compute_centers()[0],
                                              pos_y=diagram_node.compute_centers()[1],
                                              task_name=text_element)


def create_graph_from_list_nodes(dict_objects, verbose=0):
    """
    Create a DiagramGraph object with diagram objects (nodes)
    :param verbose:
    :param dict_objects:
    :return: diagram_graph
    """
    logger.info("create_graph_from_list_nodes")

    list_diagram_nodes = []
    diagram_graph = nx.Graph()
    for count, node in enumerate(dict_objects['nodes']):
        diagram_node = DiagramNode(count, node['coordinate'], node['class_shape'], node['text'])
        list_diagram_nodes.append(diagram_node)

    list_diagram_nodes_ordered = order_nodes_by_center_positions(list_diagram_nodes)

    for node_item in list_diagram_nodes_ordered:
        diagram_graph.add_node(node_item.id, data=node_item)
        if verbose:
            logger.info('DiagramNode item: %s', node_item)

    dict_connected_nodes = build_dictionary_with_connected_nodes(diagram_graph)
    list_tuples_connected_nodes = build_list_tuples_connected_nodes(dict_connected_nodes, diagram_graph)

    # diagram_graph = update_diagram_graph_with_edges(diagram_graph, list_tuples_connected_nodes)

    for tuple_node in list_tuples_connected_nodes:
        diagram_graph.add_edge(tuple_node[0], tuple_node[1])

    return diagram_graph, list_diagram_nodes_ordered


# def update_diagram_graph_with_edges(diagram_graph, list_tuples_connected_nodes):
#     for tuple_node in list_tuples_connected_nodes:
#         diagram_graph.add_edge(tuple_node[0], tuple_node[1])
#
#     return diagram_graph


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
    Build list of tuples of nodes which are connected based on a minimum distance
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
    list_nodes_sorted = sorted(list_nodes, key=lambda x: [x.y, x.x])
    list_nodes_sorted_filtered = list(filter(lambda x: x.type in list_diagram_types_allowed, list_nodes_sorted))

    return list_nodes_sorted_filtered


def transform_graph_to_bpmn(diagram_graph, list_diagram_nodes, diagram_filename_unique_id, diagram_name='diagram1'):
    """
    Build a BPMN file (.bpmn or .xml) with a dictionary of elements linked to a diagram
    :param diagram_graph: DiagramGraph object.
    :param diagram_filename_unique_id: string object.
    :param diagram_name: string object.
    :return:
    """
    logger.info("transform_graph_to_bpmn")

    bpmn_graph = BpmnE2DiagramGraph()
    bpmn_graph.create_new_diagram_graph(diagram_name=diagram_name)
    process_id = bpmn_graph.add_process_to_diagram()

    m_adj = nx.adjacency_matrix(diagram_graph)
    m_adj_dense = m_adj.todense()

    list_aux = []

    for diagram_node in list_diagram_nodes:
        print('x', diagram_node)
        [task1_id, _] = create_bpmn_graph_element(diagram_node, process_id, bpmn_graph)
        diagram_node.set_idbpmn(task1_id)
        list_aux.append(diagram_node)

    generate_layout(bpmn_graph)

    tree = BpmnDiagramGraphExport.export_xml_etree(bpmn_graph)
    xml_str = tostring(tree.getroot())
    root = etree.fromstring(xml_str)
    namespaces = root.nsmap

    # Remove all arrows
    for edge in root.findall(".//bpmndi:BPMNEdge", namespaces):
        edge.getparent().remove(edge)

    # Check ids of list and xml
    [print(node) for node in list_diagram_nodes]
    [print(node.attrib['bpmnElement']) for node in root.findall(".//bpmndi:BPMNShape", namespaces)]

    for bpmn_node in root.findall(".//bpmndi:BPMNShape", namespaces):
        for node in list_diagram_nodes:
            if node.get_idbpmn() == bpmn_node.attrib['bpmnElement']:
                bounds = bpmn_node.findall('.//omgdc:Bounds', namespaces)
                for bound in bounds:
                    bound.attrib['x'] = str(node.get_x())
                    bound.attrib['y'] = str(node.get_y())
                    bound.attrib['width'] = str(node.get_w())
                    bound.attrib['height'] = str(node.get_h())

    et = etree.ElementTree(root)
    et.write(PATH_DIR_DIAGRAMS + diagram_filename_unique_id + '.xml', pretty_print=True)


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


def do_cmd_to_shell_diagram_detector(diagram_path_filename):
    """
    Execute a cmd command and return output string
    :param diagram_path_filename: String object.
    :return: dict_objects. Dictionary object. Dictionary with diagram objects.
    """
    logger.info("do_cmd_to_shell_diagram_detector")

    script_path_filename = PATH_DIAGRAM_GURU_PROJECT + '/diagram_detector/detector_main.py'

    logger.info("path_diagram_guru_project: %s", PATH_DIAGRAM_GURU_PROJECT)

    cmd_python_version = [PATH_VENV_DIAGRAM_GURU + "python --version"]
    process = subprocess.Popen(cmd_python_version, stdout=subprocess.PIPE, stderr=None)
    cmd_output_python_version = process.communicate()

    print(cmd_output_python_version[0].decode("utf-8"))

    cmd_diagram_detector = [PATH_VENV_DIAGRAM_GURU + "python", script_path_filename,
                            "--diagram_filename", diagram_path_filename,
                            "--display_image", "True"]

    process = subprocess.Popen(cmd_diagram_detector, stdout=subprocess.PIPE, stderr=None)
    cmd_output = process.communicate()

    dict_objects_str = cmd_output[0].decode("utf-8")

    if dict_objects_str and dict_objects_str.strip():
        dict_objects_str = str(dict_objects_str).replace("'", '"')
        dict_objects = eval(dict_objects_str)
    else:
        dict_objects = {}

    return dict_objects
