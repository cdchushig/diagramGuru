import networkx as nx

DISTANCE_MIN = 2


class DiagramGraph(nx.Graph):

    def __init__(self):
        super(DiagramGraph, self).__init__()

    def add_node(self, id, node):
        super().add_node(id, data=node)


