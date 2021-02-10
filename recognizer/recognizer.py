import tkinter as tk
from tkinter import ttk
from tkinter import Checkbutton, IntVar
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
from pathlib import Path
import cv2

from graph import Graph
from codeGenerator import CodeGenerator
from text_model.text_classifier import TextClassifier
from model.shape_classifier import ShapeClassifier
from flowchart_generator.flowchart_generator import FlowchartGenerator

dir_project = Path().resolve().parent
path_project = str(dir_project)

test_path = path_project + '/recognizer/img_tests/'
img_path = test_path + 'test13.jpeg'

image = cv2.imread(img_path)

# Shape classifier
overlap_thresh_1 = 0.8
overlap_thresh_2 = 0.2
bbox_threshold = 0.43

sc_classifier = ShapeClassifier(
    path_project + "/recognizer/model/training_results/8",
    use_gpu=False,
    overlap_thresh_1=overlap_thresh_1,
    overlap_thresh_2=overlap_thresh_2,
    bbox_threshold=bbox_threshold,
    num_rois=32
)

shape_nodes = sc_classifier.predict(image, display_image=True)
print(*shape_nodes)

# Text classifier
# tc = TextClassifier()
# text_nodes = tc.recognize(img_path)
text_nodes = []

# build the graph
graph = Graph(img_path, text_nodes, shape_nodes)
flow = graph.generate_graph()

results_path = path_project + '/results'

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