import os
import numpy as np
import cv2
import ast
import random
import string
from lxml import etree
from lxml.builder import ElementMaker

from app.app_consts import Consts
from app.diagram_edge import DiagramEdge
from app.diagram_shape import DiagramShape
from app.diagram_vision import process_lines

import logging
import coloredlogs

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

DIST_MIN_EDGING = 350
NUM_RAND_DIGITS = 7

NSMAP = {
    "od": 'http://tk/schema/od',
    "odDi": 'http://tk/schema/odDi',
    "dc": 'http://www.omg.org/spec/DD/20100524/DC'
}

LIST_ALLOWED_DIAGRAM_SHAPES = [Consts.ACTOR, Consts.OVAL, Consts.DECISION, Consts.CIRCLE]
LIST_ALLOWED_DIAGRAM_EDGES = [Consts.LINE, Consts.ARROW]

PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(__file__))
PATH_DIR_DIAGRAMS = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'


def generate_random_id(size: int, chars=string.ascii_lowercase + string.digits) -> str:
    return ''.join(random.choice(chars) for _ in range(size))


def get_uid_diagram_element(pred_class_name: str) -> str:
    id_rand_node = generate_random_id(NUM_RAND_DIGITS)
    if pred_class_name == Consts.OBJECT:
        id_diagram_element = "{0}_{1}".format(Consts.OBJECT.capitalize(), id_rand_node)
    elif pred_class_name == Consts.DECISION:
        id_diagram_element = "{0}_{1}".format(Consts.DECISION.capitalize(), id_rand_node)
    elif pred_class_name == Consts.CIRCLE:
        id_diagram_element = "{0}_{1}".format(Consts.CIRCLE.capitalize(), id_rand_node)
    elif pred_class_name == Consts.OVAL:
        id_diagram_element = "{0}_{1}".format(Consts.OVAL.capitalize(), id_rand_node)
    elif pred_class_name == Consts.ACTOR:
        id_diagram_element = "{0}_{1}".format(Consts.ACTOR.capitalize(), id_rand_node)
    elif pred_class_name == Consts.LINE:
        id_diagram_element = "{0}_{1}".format(Consts.LINK.capitalize(), id_rand_node)
    elif pred_class_name == Consts.ARROW:
        id_diagram_element = "{0}_{1}".format(Consts.LINK.capitalize(), id_rand_node)
    else:
        id_diagram_element = "None"

    return id_diagram_element


def get_list_shape_edge_elements(local_diagram_img: np.array, dict_diagram_objects: dict) -> (list, list):

    list_diagram_edges = []
    list_diagram_shapes = []

    for diagram_element in dict_diagram_objects["nodes"]:
        pred_box = np.array(ast.literal_eval(diagram_element['pred_box']), dtype='int')
        pred_class_name = diagram_element['pred_class_name']
        pred_score = diagram_element['score']
        id_diagram_element = get_uid_diagram_element(pred_class_name)

        if pred_class_name in LIST_ALLOWED_DIAGRAM_SHAPES:
            diagram_shape = DiagramShape(id=id_diagram_element,
                                         bounding_box=pred_box,
                                         pred_class=pred_class_name,
                                         likelihood=pred_score,
                                         text=str(id_diagram_element))
            list_diagram_shapes.append(diagram_shape)

        if pred_class_name in LIST_ALLOWED_DIAGRAM_EDGES:

            diagram_edge = DiagramEdge(id=id_diagram_element,
                                       bounding_box=pred_box,
                                       pred_class=pred_class_name,
                                       likelihood=pred_score)

            list_diagram_edges.append(diagram_edge)

    list_diagram_edges_real = []

    for diagram_edge in list_diagram_edges:
        crop_img = get_crop_img_by_bounding_box(local_diagram_img, diagram_edge.get_bounding_box())
        k_src, k_dst = process_lines(crop_img, diagram_edge.get_bounding_box())

        if k_src is not None and k_dst is not None:
            diagram_edge.set_k_src(k_src)
            diagram_edge.set_k_dst(k_dst)
            list_diagram_edges_real.append(diagram_edge)

    list_diagram_shapes_real = list_diagram_shapes

    return list_diagram_shapes_real, list_diagram_edges_real


def get_crop_img_by_bounding_box(img, pred_box):

    x = int(pred_box[0])
    y = int(pred_box[1])
    w = int(pred_box[2] - pred_box[0])
    h = int(pred_box[3] - pred_box[1])

    crop_img = img[y:y + h, x:x + w]

    return crop_img


def compute_shortest_distance(p, a, b):
    """Cartesian distance from point to line segment
    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)


def get_list_line_segments(pred_box: np.array) -> (list, np.matrix, np.matrix):
    list_line_segments = []
    w = pred_box[2] - pred_box[0]
    h = pred_box[3] - pred_box[1]

    p1 = np.array([[pred_box[0], pred_box[1]]])
    p2 = np.array([[pred_box[0] + w, pred_box[1]]])

    list_line_segments.append((p1, p2))

    p3 = np.array([[pred_box[0] + w, pred_box[1]]])
    p4 = np.array([[pred_box[2], pred_box[3]]])

    list_line_segments.append((p3, p4))

    p5 = np.array([[pred_box[0], pred_box[1] + h]])
    p6 = np.array([[pred_box[2], pred_box[3]]])

    list_line_segments.append((p5, p6))

    p7 = np.array([[pred_box[0], pred_box[1]]])
    p8 = np.array([[pred_box[0], pred_box[1] + h]])

    list_line_segments.append((p7, p8))

    a = np.concatenate((p1, p3, p5, p7), axis=0)
    b = np.concatenate((p2, p4, p6, p8), axis=0)

    return list_line_segments, a, b


def resize_image_by_wh(original_img: np.array, new_width: int, new_height: int) -> np.array:
    original_height, original_width = original_img.shape[:2]

    if new_width is None and new_height is None:
        return original_img

    if new_width is None:
        r = new_height / float(original_height)
        new_dim = (int(original_width * r), new_height)
    else:
        r = new_width / float(original_width)
        new_dim = (new_width, int(original_height * r))

    resized_img = cv2.resize(original_img, new_dim, interpolation=cv2.INTER_AREA)

    logger.info('original image size: (%i, %i)', original_width, original_height)
    logger.info('resized image size: (%i, %i)', resized_img.shape[1], resized_img.shape[0])

    return resized_img


def get_uid_shape(diagram_edge: DiagramEdge, list_dists: list) -> (int, float):
    list_uid_src = list(zip(*list_dists))[0]
    m_dist = np.array(list(zip(*list_dists))[1])
    v_idx = np.unravel_index(np.argmin(m_dist, axis=None), m_dist.shape)
    uid_src_shape = list_uid_src[v_idx[0]]
    dst_objs = m_dist[v_idx]
    return uid_src_shape, dst_objs



def identify_src_dst_shapes(diagram_edge: DiagramEdge, list_shapes: list,
                            list_dists_src: list, list_dists_dst: list) -> (str, str):

    uid_src_shape, dist_src = get_uid_shape(diagram_edge, list_dists_src)
    uid_dst_shape, dist_dst = get_uid_shape(diagram_edge, list_dists_dst)

    # TODO check how many shapes can be matched
    shape_src = list(filter(lambda elem_shape: elem_shape.id == uid_src_shape, list_shapes))[0]
    shape_dst = list(filter(lambda elem_shape: elem_shape.id == uid_dst_shape, list_shapes))[0]

    # TODO check other types of diagrams
    if shape_src.get_pred_class() != Consts.ACTOR and shape_dst.get_pred_class() != Consts.ACTOR:
        return 0, 0

    if shape_src.get_pred_class() != Consts.ACTOR:
        shape_src_aux = shape_src
        shape_src = shape_dst
        shape_dst = shape_src_aux
        k_dst_aux = diagram_edge.get_k_dst()
        k_src_aux = diagram_edge.get_k_src()
        diagram_edge.set_k_src(k_dst_aux)
        diagram_edge.set_k_dst(k_src_aux)

    return shape_src.get_id(), shape_dst.get_id()


def is_point_in_bounding_box(bb, p):
    return bb.x1 <= p.x and p.x <= bb.x2 and bb.y1 <= p.y and p.y <= bb.y2


def resize_image_by_scale_percent(original_img: np.array, scale_percent: int) -> np.array:
    width = int(original_img.shape[1] * scale_percent / 100)
    height = int(original_img.shape[0] * scale_percent / 100)
    new_dim = (width, height)
    resized_img = cv2.resize(original_img, new_dim, interpolation=cv2.INTER_AREA)
    return resized_img


def create_list_shape_edges_objects(diagram_img: np.array, dict_diagram_objects: dict) -> (list, list):

    lst_diagram_shapes, lst_diagram_edges = get_list_shape_edge_elements(diagram_img, dict_diagram_objects)

    lst_diagram_edges_final = []

    for diagram_edge in lst_diagram_edges:

        lst_dist_src = []
        lst_dist_dst = []

        for diagram_shape in lst_diagram_shapes:
            pred_box = diagram_shape.get_bounding_box()
            list_line_segments, m_a, m_b = get_list_line_segments(pred_box)
            lst_dist_src.append((diagram_shape.get_id(), compute_shortest_distance(diagram_edge.get_k_src(), m_a, m_b)))
            lst_dist_dst.append((diagram_shape.get_id(), compute_shortest_distance(diagram_edge.get_k_dst(), m_a, m_b)))

        uid_src_shape, uid_dst_shape = identify_src_dst_shapes(diagram_edge, lst_diagram_shapes, lst_dist_src, lst_dist_dst)

        if uid_src_shape == 0 and uid_dst_shape == 0:
            print('Removed current edge')
        else:
            diagram_edge.set_s_src(uid_src_shape)
            diagram_edge.set_s_dst(uid_dst_shape)
            lst_diagram_edges_final.append(diagram_edge)

    for diagram_shape in lst_diagram_shapes:
        lst_src_shapes = list(filter(lambda elem_edge: elem_edge.get_s_src() == diagram_shape.get_id(), lst_diagram_edges_final))
        diagram_shape.set_list_edges(lst_src_shapes)

    return lst_diagram_shapes, lst_diagram_edges_final


def create_str_xml_file(list_diagram_shapes: list, list_diagram_edges: list, diagram_filename_uid: str) -> str:

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
    for diagram_shape in list_diagram_shapes:
        pred_class_name = diagram_shape.get_pred_class()
        if pred_class_name in LIST_ALLOWED_DIAGRAM_SHAPES:
            if pred_class_name == Consts.OBJECT:
                od_board.append(elem_maker_od.object(id=diagram_shape.get_id(), attributeValues=diagram_shape.get_text()))
            elif pred_class_name == Consts.DECISION:
                od_board.append(elem_maker_od.decision(id=diagram_shape.get_id(), attributeValues=diagram_shape.get_text()))
            elif pred_class_name == Consts.CIRCLE:
                od_board.append(elem_maker_od.circle(id=diagram_shape.get_id(), attributeValues=diagram_shape.get_text()))
            elif pred_class_name == Consts.OVAL:
                od_board.append(elem_maker_od.oval(id=diagram_shape.get_id(), attributeValues=diagram_shape.get_text()))
            elif pred_class_name == Consts.ACTOR:
                od_shape_aux = elem_maker_od.actor(id=diagram_shape.get_id(), attributeValues = diagram_shape.get_text())
                for link_shape in diagram_shape.get_list_edges():
                    em_link = elem_maker_od.links(link_shape.get_id())
                    od_shape_aux.append(em_link)
                od_board.append(od_shape_aux)
            else:
                id_diagram_node = "None"

            pred_box = diagram_shape.get_bounding_box()

            x = pred_box[0]
            y = pred_box[1]
            w = pred_box[2] - pred_box[0]
            h = pred_box[3] - pred_box[1]

            list_id_nodes.append((diagram_shape.get_id(), x, y, w, h, pred_class_name, 'odshape'))

    for diagram_edge in list_diagram_edges:
        pred_class_name = diagram_edge.get_pred_class()
        if pred_class_name == Consts.LINE:
            od_board.append(elem_maker_od.link(id=diagram_edge.get_id(),
                                               sourceRef=diagram_edge.get_s_src(),
                                               targetRef=diagram_edge.get_s_dst()))
        else: # Arrow
            od_board.append(elem_maker_od.link(id=diagram_edge.get_id(),
                                               sourceRef=diagram_edge.get_s_src(),
                                               targetRef=diagram_edge.get_s_dst()))

        x1 = diagram_edge.get_k_src()[0]
        y1 = diagram_edge.get_k_src()[1]
        x2 = diagram_edge.get_k_dst()[0]
        y2 = diagram_edge.get_k_dst()[1]

        list_id_nodes.append((diagram_edge.get_id(), x1, y1, x2, y2, diagram_edge.get_pred_class(), 'odlink'))

    od_di_plane = elem_maker_od_di.odPlane(id='Plane_debug', boardElement='Board_debug')
    od_di_board.append(od_di_plane)

    for d_node in list_id_nodes:
        if (d_node[5] == Consts.ARROW) or (d_node[5] == Consts.LINE):
            em_waypoint1 = elem_maker_od_di.waypoint(x=str(d_node[1]), y=str(d_node[2]))
            em_waypoint2 = elem_maker_od_di.waypoint(x=str(d_node[3]), y=str(d_node[4]))
            od_di_link = elem_maker_od_di.link(id=d_node[0] + '_di', boardElement=d_node[0])
            od_di_link.append(em_waypoint1)
            od_di_link.append(em_waypoint2)
            od_di_plane.append(od_di_link)
        else:
            em_bound = elem_maker_dc.Bounds(x=str(d_node[1]), y=str(d_node[2]), width=str(d_node[3]), height=str(d_node[4]))
            od_di_shape = elem_maker_od_di.odShape(id=d_node[0] + '_di', boardElement=d_node[0])
            od_di_shape.append(em_bound)
            od_di_plane.append(od_di_shape)

    et = etree.ElementTree(od_root)
    et.write(PATH_DIR_DIAGRAMS + diagram_filename_uid + '.xml',
             pretty_print=True,
             xml_declaration=True,
             encoding="utf-8")

    xml_str = etree.tostring(od_root, pretty_print=True, xml_declaration=True, encoding='utf-8')
    # print(etree.dump(od_root))

    return xml_str


def get_diagram_image_bytes(path_diagram_filename_uid, ext_file):

    diagram_img = cv2.imread(path_diagram_filename_uid)
    diagram_img_resized = resize_image_by_wh(diagram_img, 800, None)
    is_success, im_buf_arr = cv2.imencode(ext_file, diagram_img_resized)
    img_bytes = im_buf_arr.tobytes()

    return img_bytes, diagram_img_resized


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


