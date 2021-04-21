import tkinter as tk
from tkinter import ttk
from tkinter import Checkbutton, IntVar
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
from pathlib import Path
import cv2
import os
import argparse

from graph import Graph
# from codeGenerator import CodeGenerator
# from text_model.text_classifier import TextClassifier
from model.shape_classifier import ShapeClassifier
# from flowchart_generator.flowchart_generator import FlowchartGenerator


from pathlib import Path

PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_arguments(parser):
    parser.add_argument('--id_model', type=int, default=63)
    parser.add_argument('--file_name', type=str, default='test3.jpeg')
    return parser.parse_args()


def detect_diagram_objects(diagram_filename):

    # dir_project = Path().resolve().parent
    # path_project = str(dir_project)

    test_path = PATH_DIAGRAM_GURU_PROJECT + '/uploads/'
    img_path = test_path + diagram_filename

    image = cv2.imread(img_path)

    # Shape classifier
    overlap_thresh_1 = 0.8
    overlap_thresh_2 = 0.2
    bbox_threshold = 0.43

    sc_classifier = ShapeClassifier(
        PATH_DIAGRAM_GURU_PROJECT + "/recognizer/model/training_results/8",
        use_gpu=False,
        overlap_thresh_1=overlap_thresh_1,
        overlap_thresh_2=overlap_thresh_2,
        bbox_threshold=bbox_threshold,
        num_rois=32
    )

    shape_nodes = sc_classifier.predict(image, display_image=True)
    # print(*shape_nodes)

    for shape_node in shape_nodes:
        print(shape_node)

    # Text classifier
    # tc = TextClassifier()
    # text_nodes = tc.recognize(img_path)
    text_nodes = []

    # build the graph
    graph = Graph(img_path, text_nodes, shape_nodes)
    flow = graph.generate_graph()

    # results_path = path_project + '/results'

    # fg = FlowchartGenerator(graph,flow,results_path)
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

    file_name = args.file_name

    detect_diagram_objects(file_name)
