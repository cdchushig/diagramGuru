import numpy as np


class DiagramShape(object):

    def __init__(self, id, bounding_box, pred_class, likelihood, text):
        super(DiagramShape, self).__init__()
        self.id = id
        self.bounding_box = bounding_box
        self.pred_class = pred_class
        self.likelihood = likelihood
        self.text = text
        self.list_edges = []

    def get_bounding_box(self) -> np.array:
        return self.bounding_box

    def get_pred_class(self) -> str:
        return self.pred_class

    def get_id(self) -> str:
        return self.id

    def get_text(self) -> str:
        return self.text

    def set_list_edges(self, list_edges):
        self.list_edges = list_edges

    def __str__(self):
        return "DiagramShape(id:" + str(self.id) + " pred_class:" + self.pred_class + ")"
