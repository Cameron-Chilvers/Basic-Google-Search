class Node:
    def __init__(self, id, link_address):
        self.id = id
        self.link_address = link_address

    # returning id
    def get_id(self):
        return self.id
    
    def get_link_address(self):
        return self.link_address


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
