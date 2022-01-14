
class DiagramNode(object):

    def __init__(self, id, coords, type, text):
        super(DiagramNode, self).__init__()
        self.id = id
        self.type = type
        self.coords = coords
        self.text = text
        self.centers = self.compute_centers()

    def get_coords(self):
        return self.coords

    def get_id(self):
        return self.id

    def get_type(self):
        return self.type

    def get_text(self):
        return self.text

    def compute_centers(self):
        cx = int((self.coords[0] + self.coords[1]) / 2)
        cy = int((self.coords[2] + self.coords[3]) / 2)
        return [cx, cy]

    def __str__(self):
        return "DiagramNode(id:" + str(self.id) + ", coords:" + str(self.coords) + ", centers:" + str(self.centers) +\
               ", type: " + str(self.type) + ",text:" + str(self.text) + ")"

