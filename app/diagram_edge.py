
class DiagramEdge(object):

    def __init__(self, id, bounding_box, pred_class, likelihood, k_src, k_tgt, s_src, s_tgt, text):
        super(DiagramEdge, self).__init__()
        self.id = id
        self.bounding_box = bounding_box
        self.pred_class = pred_class
        self.likelihood = likelihood
        self.k_src = k_src
        self.k_tgt = k_tgt
        self.s_src = s_src
        self.s_tgt = s_tgt
        self.text = text

    def get_k_src(self):
        return self.k_src

    def get_k_tgt(self):
        return self.k_tgt
