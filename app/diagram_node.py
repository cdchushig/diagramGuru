
class DiagramNode(object):

    def __init__(self, id, coords, type, text):
        super(DiagramNode, self).__init__()
        self.id = id
        self.type = type
        self.coords = coords
        self.text = text
        self.centers = self.compute_centers()
        self.x = coords[0]
        self.y = coords[2]
        self.w = coords[1] - coords[0]
        self.h = coords[3] - coords[2]
        self.idbpmn = 0

    def set_idbpmn(self, id):
        self.idbpmn = id

    def get_coords(self):
        return self.coords

    def get_id(self):
        return self.id

    def get_idbpmn(self):
        return self.idbpmn

    def get_type(self):
        return self.type

    def get_text(self):
        return self.text

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_w(self):
        return self.w

    def get_h(self):
        return self.h

    def compute_centers(self):
        cx = int((self.coords[0] + self.coords[1]) / 2)
        cy = int((self.coords[2] + self.coords[3]) / 2)
        return [cx, cy]

    def __str__(self):
        return "DiagramNode(id:" + str(self.id) + ", coords:" + str(self.coords) + ", centers:" + str(self.centers) +\
               ", type: " + str(self.type) + ",text:" + str(self.text) + ")" + ",idbpmn:" + str(self.idbpmn)

