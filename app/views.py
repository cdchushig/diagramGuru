import os
import numpy as np
from datetime import datetime
import base64
import json
import requests
import subprocess
import itertools as it
import operator
import random
import string
import ast

from django.http import HttpResponseRedirect
from django.contrib.auth import authenticate, login
from django.conf import settings
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.http import HttpResponse

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from shapely.geometry import Point, LineString
from .forms import UploadFileForm, NewUserForm, NewModelForm
from .models import Diagram
from .serializers import DiagramSerializer
from .app_consts import Consts
from .diagram_node import DiagramNode
from .diagram_edge import DiagramEdge
from .diagram_shape import DiagramShape

from PIL import Image
from pdf2image import convert_from_path

import networkx as nx
import cv2

from app.diagram_utils import compute_distance_between_nodes

from lxml import etree
from lxml.builder import ElementMaker
from xml.etree.ElementTree import tostring

DIST_MIN_EDGING = 350
PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(__file__))
PATH_DIR_UPLOADS = PATH_DIAGRAM_GURU_PROJECT + "/uploads/"
PATH_DIR_DIAGRAMS = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'
PATH_DIAGRAM_GURU_DIAGRAMS = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'

PATH_VENV_DIAGRAM_GURU = ''

BPMN_MODEL_NS = 'http://www.omg.org/spec/BPMN/20100524/MODEL'
DIAG_INTERCHANGE_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
DIAG_COMMON_NS = "http://www.omg.org/spec/DD/20100524/DC"

NUM_RAND_DIGITS = 7

NSMAP = {
    "od": 'http://tk/schema/od',
    "odDi": 'http://tk/schema/odDi',
    "dc": 'http://www.omg.org/spec/DD/20100524/DC'
}

LIST_DIAGRAM_ELEMENTS = [Consts.ACTOR, Consts.OVAL, Consts.LINE, Consts.ARROW]
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
            messages.success(request, "Registration successful.")
            return redirect("/list_diagram")

        messages.error(request, "Unsuccessful registration. Invalid information.")

    form = NewUserForm()

    return render(request=request, template_name="save_model.html", context={"register_form": form})


def show_dashboard(request):
    return render(request, "dashboard.html", context={})


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
            return HttpResponseRedirect('/open_external_diagram')
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
    diagram_filename = diagram_filename_uploaded_unique_id_with_extension.split('/')[-1]
    im_b64 = base64.b64encode(img_bytes).decode("utf8")
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = json.dumps({"image": im_b64, "diagram_name": diagram_filename})
    response = requests.post(Consts.API_URL, data=payload, headers=headers)

    dict_objects = {}

    try:
        dict_objects = response.json()
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

    # dict_diagram_objects = do_cmd_to_shell_diagram_detector(diagram_path_filename_unique_id)

    diagram_img = cv2.imread(diagram_path_filename_unique_id)
    is_success, im_buf_arr = cv2.imencode(ext_file, diagram_img)
    img_bytes = im_buf_arr.tobytes()
    dict_diagram_objects = do_request_to_api_diagram_detector(img_bytes, diagram_path_filename_unique_id)
    dict_diagram_objects_complete = complete_dict_diagram_objects(dict_diagram_objects)

    if dict_diagram_objects:
        # diagram_graph, list_diagram_nodes_ordered = create_graph_from_list_nodes(dict_diagram_objects, verbose=1)
        # transform_graph_to_bpmn(diagram_graph, list_diagram_nodes_ordered, diagram_filename_unique_id)
        xml_str = create_str_xml_file(dict_diagram_objects, diagram_filename_unique_id)
    else:
        logger.error('No objects in diagram recognized!')

    return diagram_filename_unique_id


def complete_dict_diagram_objects(dict_diagram_objects):

    list_diagram_nodes = dict_diagram_objects["nodes"]
    print('holi')


def compute_shortest_distance_v2(p, a, b):
    """"
    Compute shortest distance from point to line segment
    - p: np.array of shape (x, 2)
    - a: np.array of shape (x, 2)
    - b: np.array of shape (x, 2)
    """
    # A = np.array([1, 0])
    # B = np.array([3, 0])
    # C = np.array([0, 1])

    l = LineString([a, b])
    p = Point(p)

    return l.distance(p)


def compute_shortest_distance_v1(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)


def get_list_sides_shape(shape: DiagramShape):
    print('holi')


def compute_euclidean_distance(keypoint, shape: DiagramShape) -> (float, str):

    list_sides_shape = get_list_sides_shape()
    for slide_shape in list_sides_shape:
        shortest_distance = compute_shortest_distance_v1(keypoint, slide_shape)

    return 0, 'pred_class'


def get_src_tgt_shapes(edge_candidate: DiagramEdge) -> (DiagramShape, DiagramShape):
    k_src = edge_candidate.get_k_src()  # (x1, y1)
    k_tgt = edge_candidate.get_k_tgt()  # (x2, y2)

    s_src = DiagramShape()
    s_tgt = DiagramShape()

    return s_src, s_tgt


def generate_random_id(size: int, chars=string.ascii_lowercase + string.digits) -> str:
    return ''.join(random.choice(chars) for _ in range(size))


def create_str_xml_file(dict_nodes: dict, diagram_filename_unique_id: str) -> str:

    # Add both namespaces and nsmap
    elem_maker_od = ElementMaker(namespace=NSMAP["od"], nsmap=NSMAP)
    elem_maker_od_di = ElementMaker(namespace=NSMAP["odDi"], nsmap=NSMAP)
    elem_maker_dc = ElementMaker(namespace=NSMAP["dc"], nsmap=NSMAP)

    # Create root element
    od_root = elem_maker_od.definitions()

    # Create both od and odDi elements
    od_board = elem_maker_od.odBoard(id='Board_debug')
    od_di_board = elem_maker_od_di.odRootBoard(id='RootBoard_debug')

    # Link od and odDi to root
    od_root.append(od_board)
    od_root.append(od_di_board)

    # Add nodes to xml
    list_id_nodes = []
    for diagram_node in dict_nodes["nodes"]:
        if diagram_node["pred_class_name"] in LIST_DIAGRAM_ELEMENTS:
            id_rand_node = generate_random_id(NUM_RAND_DIGITS)
            if diagram_node["pred_class_name"] == Consts.OBJECT:
                id_diagram_node = "{0}_{1}".format(Consts.OBJECT.capitalize(), id_rand_node)
                od_board.append(elem_maker_od.object(id=id_diagram_node))
            elif diagram_node["pred_class_name"] == Consts.DECISION:
                id_diagram_node = "{0}_{1}".format(Consts.DECISION.capitalize(), id_rand_node)
                od_board.append(elem_maker_od.decision(id=id_diagram_node))
            elif diagram_node["pred_class_name"] == Consts.CIRCLE:
                id_diagram_node = "{0}_{1}".format(Consts.CIRCLE.capitalize(), id_rand_node)
                od_board.append(elem_maker_od.circle(id=id_diagram_node))
            elif diagram_node["pred_class_name"] == Consts.OVAL:
                id_diagram_node = "{0}_{1}".format(Consts.OVAL.capitalize(), id_rand_node)
                od_board.append(elem_maker_od.oval(id=id_diagram_node))
            elif diagram_node["pred_class_name"] == Consts.ACTOR:
                id_diagram_node = "{0}_{1}".format(Consts.ACTOR.capitalize(), id_rand_node)
                od_board.append(elem_maker_od.actor(id=id_diagram_node))
            elif diagram_node["pred_class_name"] == Consts.LINE:
                id_diagram_node = "{0}_{1}".format(Consts.LINK.capitalize(), id_rand_node)
                od_board.append(elem_maker_od.link(id=id_diagram_node))
            elif diagram_node["pred_class_name"] == Consts.ARROW:
                id_diagram_node = "{0}_{1}".format(Consts.LINK.capitalize(), id_rand_node)
                od_board.append(elem_maker_od.link(id=id_diagram_node))
            else:
                id_diagram_node = "None"
            pred_box = ast.literal_eval(diagram_node["pred_box"])


            x = pred_box[0]
            y = pred_box[1]
            w = pred_box[2] - pred_box[0]
            h = pred_box[3] - pred_box[1]

            list_id_nodes.append((id_diagram_node, x, y, w, h, diagram_node["pred_class_name"]))

    od_di_plane = elem_maker_od_di.odPlane(id='Plane_debug', boardElement='Board_debug')
    od_di_board.append(od_di_plane)

    for d_node in list_id_nodes:
        if (d_node[5] == Consts.ARROW) or (d_node[5] == Consts.LINE):
            print('holi')
        else:
            em_bound = elem_maker_dc.Bounds(x=str(d_node[1]), y=str(d_node[2]), width=str(d_node[3]), height=str(d_node[4]))
            od_di_shape = elem_maker_od_di.odShape(id=d_node[0] + '_di', boardElement=d_node[0])
            od_di_shape.append(em_bound)
            od_di_plane.append(od_di_shape)

    et = etree.ElementTree(od_root)
    et.write(PATH_DIR_DIAGRAMS + diagram_filename_unique_id + '.xml',
             pretty_print=True,
             xml_declaration=True,
             encoding="utf-8")

    xml_str = etree.tostring(od_root, pretty_print=True, xml_declaration=True, encoding='utf-8')
    # print(etree.dump(od_root))

    return xml_str


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


def create_model(request):
    bpmn_filename = os.path.join(settings.BASE_DIR, 'static', 'app', 'diagrams', 'blank-diagram.xml')
    with open(bpmn_filename, 'r') as f:
        bpmn_file_content = f.read()
    context = {'bpmn_filename': bpmn_filename, 'bpmn_file_content': bpmn_file_content, 'id_bpmn': -1}
    template = 'modeler/modeler_oc.html'
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


def open_external_diagram(request):
    bpmn_filename_xml = request.session.get('diagram_name')
    bpmn_filename = PATH_DIAGRAM_GURU_DIAGRAMS + bpmn_filename_xml + '.xml'

    with open(bpmn_filename, 'r') as f:
        bpmn_file_content = f.read()

    context = {'bpmn_filename': bpmn_filename, 'bpmn_file_content': bpmn_file_content, 'id_bpmn': -1}
    template = 'modeler/modeler_oc.html'
    return render(request, template, context)

    # tree = etree.parse(bpmn_path_filename_xml)
    # xml_str = etree.tostring(tree.getroot())

    # context = {'bpmn_file_content': xml_str, 'id_bpmn': 1}
    # logger.info('Loaded xml file: %s', request.session.get('diagram_name'))
    # return render(request, 'modeler_oc.html', context)


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

    cmd_diagram_detector = [PATH_VENV_DIAGRAM_GURU, script_path_filename,
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

