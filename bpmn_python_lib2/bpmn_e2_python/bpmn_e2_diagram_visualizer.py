# coding=utf-8
"""
BPMN diagram visualization methods
"""

def make_node_attributes(node): 
    attribute_str = ""
    label = node[1]['node_name']
    type = node[1]['type']
    
    attribute_str += f'label="{label}" '

    if type == "exclusiveGateway": 
        attribute_str += f'shape="diamond"'
    else: 
        attribute_str += f'shape="box" '

    return attribute_str

def make_node(node): 
    dot_str = ""

    dot_str += f"{node[0]} ["
    dot_str += make_node_attributes(node)

    dot_str += "]\n"

    return dot_str

def make_edge(edge, associations, sequence_flows): 
    dot_str = ""
    sourceRef = edge[2]['sourceRef']
    targetRef = edge[2]['targetRef']
    print(associations.values())

    exists_association = list((filter(lambda x: x['sourceRef'] in sequence_flows \
        and sequence_flows[x['sourceRef']]['sourceRef'] == sourceRef \
        and sequence_flows[x['sourceRef']]['targetRef'] == targetRef, associations.values())))
    
    if (len(exists_association) == 0):
        dot_str += f"{sourceRef} -> {targetRef}"
        dot_str += "\n"
    else:
        print('hello')
    
    return dot_str   

def begin_dot(file): 
    begin = "digraph G {\n rankdir=LR\ngraph [splines=ortho]\n"
    file.write(begin)

def make_activity_duration(activity_duration):
    key = activity_duration[0]
    value = activity_duration[1]
    dot_str = ""

    dot_str += f'{key} [color="green"'
    """ TODO attributes """
    dot_str += "]"

    return dot_str

def make_monitoring_group(monitoring_group):
    key = monitoring_group[0]
    value = monitoring_group[1]
    dot_str = ""

    dot_str += f'{key} [color="red"'
    """ TODO attributes """
    dot_str += "]"

    return dot_str

def make_decision_points(decision_point):
    key = decision_point[0]
    value = decision_point[1]
    dot_str = ""

    dot_str += f'{key} [color="blue"'
    """ TODO attributes """
    dot_str += "]\n"

    return dot_str



def make_association(association, sequence_flows):
    key = association[0]
    value = association[1]
    s_flow = sequence_flows.get(value["sourceRef"], None)
    dot_str = ""

    dot_str += f'{value["sourceRef"]} -> {value["targetRef"]} ['
    """ TODO attributes """
    dot_str += "]\n"

    if (s_flow):
        dot_str += f'{value["sourceRef"]} [width=0.01,height=0.01, shape=none, label=""]\n'
        dot_str += f'{s_flow["sourceRef"]} -> {value["sourceRef"]} [arrowhead=none] \n'
        dot_str += f'{value["sourceRef"]} -> {s_flow["targetRef"]}'
        
        
    dot_str += f'{{rank= same; {value["sourceRef"]};{value["targetRef"]}}}'

    return dot_str

def fill_dot(file, bpmn_e2_diagram): 
    nodes = bpmn_e2_diagram.diagram_graph.nodes().data()
    edges = bpmn_e2_diagram.diagram_graph.edges().data()

    activity_durations = bpmn_e2_diagram.activity_durations
    monitoring_groups = bpmn_e2_diagram.monitoring_groups
    decision_points = bpmn_e2_diagram.decision_points
    associations = bpmn_e2_diagram.associations

    sequence_flows = bpmn_e2_diagram.sequence_flows

    for node in nodes: 
        dot_node = make_node(node) 
        file.write(dot_node) 

    for edge in edges: 
        dot_edge = make_edge(edge, associations, sequence_flows)
        file.write(dot_edge)

    for activity_duration in activity_durations.items(): 
        dot_duration = make_activity_duration(activity_duration)
        file.write(dot_duration)

    for monitoring_group in monitoring_groups.items(): 
        dot_monitoring_group = make_monitoring_group(monitoring_group)
        file.write(dot_monitoring_group)

    for decision_point in decision_points.items(): 
        dot_decision_point = make_decision_points(decision_point)
        file.write(dot_decision_point)

    for association in associations.items(): 
        dot_association = make_association(association, sequence_flows)
        file.write(dot_association)


def end_dot(file):
    end = "}"
    file.write(end)

def diagram_to_dot(bpmn_e2_diagram, filename): 
    file = open(filename + '.dot', 'w')

    begin_dot(file)

    fill_dot(file, bpmn_e2_diagram)

    end_dot(file)

    file.close()