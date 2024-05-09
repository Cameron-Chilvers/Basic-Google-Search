from Node_File import Node, NodeMap
from Graph import Graph
import glob

# create a graph
# create a dict of nodes

# Loop through url_id_mapping to create the nodes

# loop through url_grpah_file / url_with_reditects to create the grpah
def test_dump():
    with open(r"C:\Users\camch\Documents\NetworkAssigment3\Webb_Spam_Corpus_graph_files\Webb_Spam_Corpus_graph_files\host_id_mapping", "r") as f:
        for i,line in enumerate(f.readlines()):
            print(line)
            data = line.strip().split(" ")

            node = Node()
            node.initalise_node(data[0], data[1])
            node.dump_node("json_dumps\\nodes\\")
            if i == 10:
                break

def test_read():
    for file in glob.glob("json_dumps\\nodes\\*.json"):
        print(file)

        node = Node()
        node.read_from_json(file)
        node.print_node()


test_read()