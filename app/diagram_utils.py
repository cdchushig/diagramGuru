import math


def compute_distance_between_nodes(node_a, node_b):
    """
    Compute the distance between two nodes
    :param node_a:
    :param node_b:
    :return:
    """
    # [x1, x2, y1, y2]
    cx1, cy1 = node_a.compute_centers()
    cx2, cy2 = node_b.compute_centers()

    return math.sqrt(math.pow(cx1 - cx2, 2) + math.pow(cy1 - cy2, 2))

