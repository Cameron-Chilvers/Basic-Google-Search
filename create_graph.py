from Node_File import Node, NodeMap
from Graph import Graph
import glob
import json 

NODE_PATH = "json_dumps\\nodes\\"

# loop through links to create the grpah
def create_graph():
    graph = Graph()

    with open(r"nodes-links\links.txt", "r") as f:
        links = f.readlines()

    for val in links:
        # Cleaning the values
        val = val.strip()
        link_split = val.split(' ')

        # Converting to int
        link_split[0] = int(link_split[0])
        link_split[1] = int(link_split[1])

        # Adding to the graph
        graph.add_to_graph(link_split[0])
        graph.add_to_graph(link_split[1])

        # Adding the link
        graph.add_edge(link_split[0], link_split[1])
        
        #print(graph.graph_matrix)

    graph.dump_graph(r"json_dumps\graph\graph")

#create_graph()

# Loading grpah from the json file
def load_graph():
    graph = Graph()

    graph.load_graph_from_json(r"json_dumps\graph\graph.json")

    return graph

# graph = load_graph()
# print(graph.graph_matrix)

# Loop through hosts to create the nodes
def create_nodes():
    with open(r"nodes-links\hosts.txt", 'r') as f:
        nodes = f.readlines()

    #print(nodes[5:])
    node_list = []
    for node in nodes:
        # Cleaning the values
        node = node.strip()

        node_split = node.split(" ")

        node_split[0] = int(node_split[0])
        node_split[1] = node_split[1]

        node = Node()
        node.initalise_node(node_split[0], node_split[1])

        node_list.append(node)
        #node.dump_node(NODE_PATH)

    return node_list

node_list = create_nodes()

def dump_nodes(node_list):
    node_dict = dict()
    for node in node_list:
        node_prop = node.create_prop_dict()

        node_dict[node_prop["id"]] = node_prop

    with open(NODE_PATH + "nodes.json", "w") as f:
        json.dump(node_dict, f, indent=5)
        

dump_nodes(node_list)


def read_nodes():
    for file in glob.glob(NODE_PATH + "*.json"):
        node = Node()

        node.load_from_json(file)

        print(node.print_node())

#read_nodes()