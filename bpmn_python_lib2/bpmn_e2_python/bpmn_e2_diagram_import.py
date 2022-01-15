# coding=utf-8
"""
Package provides functionality for importing from BPMN-E2 XML to graph representation
"""
from xml.dom import minidom

import bpmn_python_lib2.bpmn_python.bpmn_import_utils as utils
import bpmn_python_lib2.bpmn_python.bpmn_python_consts as consts
import bpmn_python_lib2.bpmn_python.bpmn_diagram_import as bpmn_import


class BpmnE2DiagramGraphImport(object):
    """
    Class BPMNDiagramGraphImport provides methods for importing BPMN-E2 XML file.
    As a utility class, it only contains static methods. This class is meant to be used from BPMNE2DiagramGraph class.
    """

    def __init__(self):
        pass

    @staticmethod
    def load_diagram_from_xml(filepath, bpmn_e2_diagram):
        """
        Reads an XML file from given filepath and maps it into inner representation of BPMN-E2 diagram.
        Returns an instance of BPMNE2DiagramGraph class.

        :param filepath: string with output filepath,
        :param bpmn_e2_diagram: an instance of BpmnE2DiagramGraph class.
        """

        """ Normal BPMN import """
        bpmn_import.BpmnDiagramGraphImport.load_diagram_from_xml(filepath, bpmn_e2_diagram)

        """ 
        TODO extend the above representation with e2 elements 
        """

        diagram_graph = bpmn_e2_diagram.diagram_graph
        activity_durations = bpmn_e2_diagram.activity_durations
        monitoring_groups = bpmn_e2_diagram.monitoring_groups
        activity_effects = bpmn_e2_diagram.activity_effects
        decision_points = bpmn_e2_diagram.decision_points
        associations = bpmn_e2_diagram.associations
        
        document = BpmnE2DiagramGraphImport.read_xml_file(filepath)
        extension_elements = document.getElementsByTagNameNS('*', 'extensionElements')[0]

        BpmnE2DiagramGraphImport.import_activity_durations(extension_elements, activity_durations)
        BpmnE2DiagramGraphImport.import_decision_points(extension_elements, decision_points)
        BpmnE2DiagramGraphImport.import_monitoring_groups(extension_elements, monitoring_groups)
        BpmnE2DiagramGraphImport.import_associations(extension_elements, associations)
        BpmnE2DiagramGraphImport.update_graph(bpmn_e2_diagram, associations)

    @staticmethod
    def update_graph_node(diagram_graph, association): 
        node = diagram_graph.nodes[association['sourceRef']]

        if 'extensionElements' in node:
            node['extensionElements'].append(association['targetRef'])
        else: 
            node['extensionElements'] = []
            node['extensionElements'].append(association['targetRef'])

    @staticmethod 
    def update_graph_edge(diagram_graph, sequence_flows, association):
        sourceRef = sequence_flows[association['sourceRef']]['sourceRef']
        targetRef = sequence_flows[association['sourceRef']]['targetRef']

        edge = diagram_graph.edges[sourceRef, targetRef]
        edge['decisionPoint'] = association['targetRef']

    @staticmethod 
    def update_graph(diagram, associations): 
        diagram_graph = diagram.diagram_graph 
        sequence_flows = diagram.sequence_flows

        for association in associations.values(): 
            try: 
                BpmnE2DiagramGraphImport.update_graph_node(diagram_graph, association)
            except: 
                BpmnE2DiagramGraphImport.update_graph_edge(diagram_graph, sequence_flows, association)




    @staticmethod
    def import_distribution_params(time_expected_dict, id, params) :
    
        for param in params: 
            name = param.getAttribute('name')
            value = param.getAttribute('value')
            unit = param.getAttribute('unit')

            time_expected_dict[id]['params'][name] = {
                'value': value,
                'unit': unit
            }


    @staticmethod
    def import_activity_durations(extension_elements, activity_durations):
        time_expected_xml = extension_elements.getElementsByTagNameNS('*', 'timeExpected')
        time_expected_dict = {}
        
        for time_expected in time_expected_xml:
            id = time_expected.getAttribute('id')
            description = time_expected.getAttribute('description')
            distribution = time_expected.getAttribute('probabilisticDistribution')
            params = time_expected.getElementsByTagNameNS('*', 'param')

            time_expected_dict[id] = {
                'description': description, 
                'distribution': distribution,
                'params': {}
            }

            BpmnE2DiagramGraphImport.import_distribution_params(time_expected_dict, id, params)
            
        for key in time_expected_dict.keys():
            activity_durations[key] = time_expected_dict[key]

    

    @staticmethod
    def import_associations(extension_elements, associations): 
        associations_xml = extension_elements.getElementsByTagNameNS('*', 'association')
        associations_dict = {}

        for association in associations_xml:
            id = association.getAttribute('id')
            sourceRef = association.getAttribute('sourceRef')
            targetRef = association.getAttribute('targetRef')
            direction = association.getAttribute('associationDirection')

            associations_dict[id] = {
                'sourceRef': sourceRef,
                'targetRef': targetRef,
                'associationDirection': direction
            }
            
        for key in associations_dict:
            associations[key] = associations_dict[key]

    @staticmethod
    def import_monitoring_variables(monitoring_groups_dict, id, monitoring_variables):
        
        for monitoring_variable in monitoring_variables: 
            variableId = monitoring_variable.getAttribute('id')
            description = monitoring_variable.getAttribute('description')
            required = monitoring_variable.getAttribute('required')
            content = monitoring_variable.getAttribute('content')

            monitoring_groups_dict[id]['monitoringVariables'][variableId] = {
                'description': description,
                'required': required,
                'content': content
            }

    @staticmethod
    def import_monitoring_groups(extension_elements, monitoring_groups):
        monitoring_groups_xml = extension_elements.getElementsByTagNameNS('*', 'monitoringGroup')
        monitoring_groups_dict = {}

        for monitoring_group in monitoring_groups_xml: 
            id = monitoring_group.getAttribute('id')
            description = monitoring_group.getAttribute('description')
            monitoring_variables = monitoring_group.getElementsByTagNameNS('*', 'monitoringVariable')

            monitoring_groups_dict[id] = {
                'description': description,
                'monitoringVariables': {}
            }

            BpmnE2DiagramGraphImport.import_monitoring_variables(monitoring_groups_dict, id, monitoring_variables)

        for key in monitoring_groups_dict.keys():
            monitoring_groups[key] = monitoring_groups_dict[key]


    @staticmethod 
    def import_decision_points(extension_elements, decision_questions):
        decision_questions_xml = extension_elements.getElementsByTagNameNS('*', 'decisionQuestion')
        decision_questions_dict = {}

        for decision_question in decision_questions_xml: 
            id = decision_question.getAttribute('id')
            description = decision_question.getAttribute('description')
            condition = decision_question.getAttribute('condition')

            decision_questions_dict[id] = {
                'description': description,
                'condition': condition
            } 

        for key in decision_questions_dict.keys():
            decision_questions[key] = decision_questions_dict[key]


    @staticmethod
    def read_xml_file(filepath):
        """
        Reads BPMN 2.0 XML file from given filepath and returns xml.dom.xminidom.Document object.

        :param filepath: filepath of source XML file.
        """
        dom_tree = minidom.parse(filepath)
        return dom_tree
