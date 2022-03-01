import numpy as np


class DiagramEdge(object):

    def __init__(self, id, bounding_box, pred_class, likelihood):
        super(DiagramEdge, self).__init__()
        self.id = id
        self.bounding_box = bounding_box
        self.pred_class = pred_class
        self.likelihood = likelihood
        self.k_src = None
        self.k_dst = None
        self.s_src = None
        self.s_dst = None
        self.text = None

    def get_pred_class(self) -> str:
        return self.pred_class

    def get_k_src(self) -> np.array:
        return self.k_src

    def get_k_dst(self) -> np.array:
        return self.k_dst

    def get_pred_class(self) -> str:
        return self.pred_class

    def get_id(self) -> str:
        return self.id

    def get_s_src(self) -> str:
        return self.s_src

    def get_s_dst(self) -> str:
        return self.s_dst

    def set_s_src(self, uid_shape: str):
        self.s_src = uid_shape

    def set_s_dst(self, uid_shape: str):
        self.s_dst = uid_shape

    def get_bounding_box(self) -> np.array:
        return self.bounding_box

    def set_k_src(self, k_src):
        self.k_src = k_src

    def set_k_dst(self, k_dst):
        self.k_dst = k_dst

    def __str__(self):
        return "DiagramEdge(id:" + str(self.id) + " pred_class:" + self.pred_class + \
               " pred_box: " + str(self.bounding_box) + \
               " k_src: " + str(self.k_src) + " k_dst:" + str(self.k_dst) + \
               " s_src: " + self.s_src + " s_dst: " + self.s_dst + ")"
