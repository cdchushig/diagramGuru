# coding=utf-8

import filecmp
import os

import bpmn_e2_python.bpmn_e2_diagram_rep as diagram
import bpmn_python.bpmn_diagram_layouter as layouter

input_directory = "images_input/"
output_directory = "images_output/"

process = 'test'

bpmn_graph = diagram.BpmnE2DiagramGraph()
bpmn_graph.load_diagram_from_csv_file(os.path.abspath(input_directory + process + ".csv"))
bpmn_graph.export_csv_file(output_directory, process + ".csv")
cmp_result = filecmp.cmp(input_directory + process + ".csv", output_directory, process + ".csv")
bpmn_graph.export_xml_file_no_di(output_directory, process + ".bpmn")

output_file = "test02.xml"
bpmn_graph = diagram.BpmnE2DiagramGraph()
bpmn_graph.create_new_diagram_graph(diagram_name="diagram1")
process_id = bpmn_graph.add_process_to_diagram()
[start_id, _] = bpmn_graph.add_start_event_to_diagram(process_id, start_event_name="start_event")
[task1_id, _] = bpmn_graph.add_task_to_diagram(process_id, task_name="task1")
bpmn_graph.add_sequence_flow_to_diagram(process_id, start_id, task1_id, "start_to_one")

[task2_id, _] = bpmn_graph.add_task_to_diagram(process_id, task_name="task2")
[end_id, _] = bpmn_graph.add_end_event_to_diagram(process_id, end_event_name="end_event")
bpmn_graph.add_sequence_flow_to_diagram(process_id, task1_id, task2_id, "one_to_two")
bpmn_graph.add_sequence_flow_to_diagram(process_id, task2_id, end_id, "two_to_end")

layouter.generate_layout(bpmn_graph)
bpmn_graph.export_xml_file(output_directory, output_file)