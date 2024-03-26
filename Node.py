class Node:
    def __init__(self, id, link_address):
        self.id = id
        self.link_address = link_address

    # returning id
    def get_id(self):
        return self.id
    
    def get_link_address(self):
        return self.link_address