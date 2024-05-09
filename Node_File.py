import json


class Node:
    def __init__(self):
        self.id = None
        self.link_address = None
        self.html = None

    def initalise_node(self, id, link_address):
        self.id = id
        self.link_address = link_address

    def add_html(self, html):
        self.html = html

    # returning id
    def get_id(self):
        return self.id
    
    def get_link_address(self):
        return self.link_address


    # DONT USE
    def dump_node(self, path):
        # id of node in json and inside is the metadata
        with open(path + str(self.id)+ ".json", "w") as f:
            json.dump(self.create_prop_dict(), f, indent=5)

    def create_prop_dict(self):
        properties = dict()
        properties['id'] = self.id
        properties['link_address'] = self.link_address
        #properties['html'] = self.html

        return properties
    
    def load_from_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f) 
        
        self.link_address = data['link_address']
        self.id = data['id']
        #self.html = data['html']
    
    def print_node(self):
        print("Node ID: %s\nNode Link %s\n" % (self.get_id(), self.get_link_address()))


class NodeMap:
    def __init__(self):
        self.node_map = dict()
    
    # Adding to the node map
    def add_to_list(self, node: Node):
        # Check for if node in there
        if node.get_id() not in self.node_map:
            self.node_map[node.get_id()] = node
    
    # Returning the Node or None
    def get_node(self, node_id):
        if node_id in self.node_map:
            return self.node_map[node_id]

        return None
    
    # Checking id node in node list
    def contains_node(self, node_id):
        return node_id in self.node_map
