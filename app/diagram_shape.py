
class DiagramShape(object):

    def __init__(self, id, bounding_box, pred_class, likelihood, k_src, k_tgt, s_src, s_tgt, text):
        super(DiagramShape, self).__init__()
        self.id = id
        self.bounding_box = bounding_box
        self.pred_class = pred_class
        self.likelihood = likelihood
        self.text = text
