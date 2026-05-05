class LongTermMemory:
    def __init__(self):
        self.memory = {}

    def create_node(self, node_id, data):
        """Create a new node in long-term memory."""
        self.memory[node_id] = {'data': data, 'links': []}

    def create_link(self, from_node_id, to_node_id):
        """Create a link between two nodes."""
        if from_node_id in self.memory and to_node_id in self.memory:
            self.memory[from_node_id]['links'].append(to_node_id)

    def retrieve(self, node_id):
        """Retrieve a node by its ID."""
        return self.memory.get(node_id)

    def access_node(self, node_id):
        """Access the data of a specific node."""
        node = self.retrieve(node_id)
        return node['data'] if node else None

    def apply_forgetting(self, node_id):
        """Remove a node from memory."""
        if node_id in self.memory:
            del self.memory[node_id]

    def retrieve_with_anchors(self, node_id):
        """Retrieve a node along with its linked nodes."""
        node = self.retrieve(node_id)
        if node:
            linked_data = [self.retrieve(linked_id) for linked_id in node['links']]
            return node, linked_data
        return None, None