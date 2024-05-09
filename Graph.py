import json
from collections import OrderedDict

class Graph:
    def __init__(self):
        self.graph_matrix = OrderedDict()

    def add_to_graph(self, id):
        # Adding to graph martix is not in there
        if id not in self.graph_matrix:
            self.graph_matrix[id] = OrderedDict()
    
    def add_edge(self, start_id, end_id, isDirected = True):
        # Checking if its actually in the matrix
        if start_id not in self.graph_matrix or end_id not in self.graph_matrix:
            print(start_id, " or ", end_id, " not found in graph")

            # can make a dump here
            return
        
        # Adding to set
        self.graph_matrix[start_id][end_id] = 0

        # If connection is both ways add it here
        if not isDirected: 
            self.graph_matrix[end_id][start_id] = 0
    
    # Returns an ids edges
    #   returns empty set if id not found
    def get_out_edges(self, id):
        if id in self.graph_matrix:
            return self.graph_matrix[id]
        
        return {}

    # Dumping as json to make it portable
    def dump_graph(self, filename):
        with open(filename + ".json", "w") as json_f:
            json.dump(self.graph_matrix, json_f, indent=5)

    # Creates graph json
    def load_graph_from_json(self, json_path):
        with open(json_path,) as f:
            self.graph_matrix = json.load(f)

