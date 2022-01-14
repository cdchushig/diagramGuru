import cv2
import argparse
from pathlib import Path

from graph import Graph
from text_model.text_classifier import TextClassifier
from model.shape_classifier import ShapeClassifier
from detector_consts import Consts

# from diagram_detector.detector_consts import Consts
# from diagram_detector.model.shape_classifier import ShapeClassifier
# from diagram_detector.text_model.text_classifier import TextClassifier
# from diagram_detector.graph import Graph

dir_project = Path().resolve()
PATH_DIAGRAM_GURU_PROJECT = str(dir_project)
PATH_DIAGRAM_UPLOADS = PATH_DIAGRAM_GURU_PROJECT + '/uploads/'
RESULTS_PATH = '/results'

import logging
import coloredlogs
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--id_model', type=int, default=63)
    parser.add_argument('--diagram_filename', type=str, default='test3.jpeg')
    parser.add_argument('--display_image', type=bool, default=False)
    return parser.parse_args()


def detect_diagram_objects(diagram_path_filename, display_image):
    logging.info(diagram_path_filename)
    path_trained_model = PATH_DIAGRAM_GURU_PROJECT + "/diagram_detector/model/training_results/8"
    # path_trained_model = PATH_DIAGRAM_GURU_PROJECT + "/model/training_results/8"

    diagram_image = cv2.imread(diagram_path_filename)

    if diagram_image is not None:
        sc_classifier = ShapeClassifier(
            path_trained_model,
            use_gpu=Consts.use_gpu,
            overlap_thresh_1=Consts.overlap_thresh_1,
            overlap_thresh_2=Consts.overlap_thresh_2,
            bbox_threshold=Consts.bbox_threshold,
            num_rois=Consts.num_rois
        )

        shape_nodes = sc_classifier.predict(diagram_image, display_image)
        list_dict_nodes = []
        for shape_node in shape_nodes:
            list_dict_nodes.append(vars(shape_node))
            # print(shape_node)

        dict_objects = {'nodes': list_dict_nodes}

        print(dict_objects)

        # tc = TextClassifier()
        # text_nodes_zip, text_nodes = tc.recognize(diagram_path_filename)

        # for text_diagram in text_nodes:
        #     print(text_diagram)

        # graph = Graph(text_nodes, shape_nodes)
        # flow = graph.generate_graph()


        # results_path = __get_results_path()

        # results_path = path_project + '/results'

        # fg = FlowchartGenerator(graph, flow, results_path)
        # fg.generate_flowchart()

        #call function to traslate to code and flowchart
        # results_path = self.__get_results_path()
        # os.mkdir(self.RESULTS_PATH+results_path)
        # cg = CodeGenerator(graph,results_path)
        # cg.generate(0,-1)
        # fg = FlowchartGenerator(graph,flow,results_path)
        # fg.generate_flowchart()
        # self.show_results(results_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    diagram_path_filename = args.diagram_filename
    # diagram_path_filename = PATH_DIAGRAM_UPLOADS + 'test04.png'

    detect_diagram_objects(diagram_path_filename, display_image=False)

