# coding=utf-8
"""
Package with BPMNDiagramGraph - graph representation of BPMN diagram
"""
import uuid

import networkx as nx

import bpmn_python.bpmn_diagram_exception as bpmn_exception
import bpmn_python.bpmn_diagram_export as bpmn_export
import bpmn_python.bpmn_diagram_import as bpmn_import
import bpmn_python.bpmn_process_csv_export as bpmn_csv_export
import bpmn_python.bpmn_process_csv_import as bpmn_csv_import
import bpmn_python.bpmn_python_consts as consts

import bpmn_python.bpmn_diagram_rep as bpmn_rep

import bpmn_e2_python.bpmn_e2_diagram_import as bpmn_e2_import


class BpmnE2DiagramGraph(bpmn_rep.BpmnDiagramGraph):
    """
    Class BPMNE2DiagramGraph implements simple inner representation of BPMN-E2 diagram,
    based on NetworkX graph implementation

    Fields:

    * diagram_graph - networkx.Graph object, stores elements of BPMN diagram as nodes. Each edge of graph represents
        sequenceFlow element. Edges are identified by IDs of nodes connected by edge. IDs are passed as edge parameters,
    * sequence_flows - dictionary (associative list) of sequence flows existing in diagram.
        Key attribute is sequenceFlow ID, value is a dictionary consisting three key-value pairs: "name" (sequence flow
        name), "sourceRef" (ID of node, that is a flow source) and "targetRef" (ID of node, that is a flow target),
    * collaboration - a dictionary that contains two dictionaries:
        * "messageFlows" - dictionary (associative list) of message flows existing in diagram. Key attribute is
            messageFlow ID, value is a dictionary consisting three key-value pairs: "name" (message flow name),
        * "sourceRef" (ID of node, that is a flow source) and "targetRef" (ID of node, that is a flow target),
        * "participants" - dictionary (associative list) of participants existing in diagram. Key attribute is
            participant ID, value is a dictionary consisting participant attributes,
    * process_elements_dictionary - dictionary that holds attribute values for imported 'process' elements. 
        Key is an ID of process, value is a dictionary of all process attributes,

    * diagram_attributes - dictionary that contains BPMN diagram element attributes,
    * plane_attributes - dictionary that contains BPMN plane element attributes.

    TODO add docs for BPMN-E2 elements
    """

    # String "constants" used in multiple places
    id_prefix = "id"
    bpmndi_namespace = "bpmndi:"

    def __init__(self):
        super().__init__()

        """
        BPMN-E2 new elements
        """
        self.activity_durations = {}
        self.monitoring_groups = {}
        self.activity_effects = {}
        self.decision_points = {}
        self.associations = {}



    def load_diagram_from_xml_file(self, filepath):
        """
        Reads an XML file from given filepath and maps it into inner representation of BPMN-E2 diagram.
        Returns an instance of BPMNE2DiagramGraph class.

        :param filepath: string with output filepath.
        """

        bpmn_e2_import.BpmnE2DiagramGraphImport.load_diagram_from_xml(filepath, self)
