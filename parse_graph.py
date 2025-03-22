def parse_graph_file(filename):
    # Define keys for which we want specific numeric conversions.
    numeric_float_keys = {"Node execution time"}
    numeric_int_keys = {
        "Node memory consumption",
        "First use of this activation",
        "Last use of this activation"
    }
    
    nodes = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    in_node_section = False  # Start processing after header
    current_node = {}

    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue

        # Start processing after encountering the torch.fx graph list header
        if not in_node_section:
            if "<torch.fx.graph._node_list object at" in line:
                in_node_section = True
            continue

        # Check for block separator, indicating end of the current node block.
        if line.startswith('----------------------------------------'):
            if current_node:
                nodes.append(current_node)
                # Stop processing if the node's name is "output"
                if current_node.get('Node name') == 'output':
                    break
                current_node = {}
            continue

        # Process valid lines in the "Key: Value" format.
        if ': ' in line:
            key, value = line.split(': ', 1)
            if key in numeric_float_keys:
                try:
                    current_node[key] = float(value)
                except ValueError:
                    current_node[key] = value
            elif key in numeric_int_keys:
                try:
                    # Convert to float first then int to handle decimal numbers.
                    current_node[key] = int(float(value))
                except ValueError:
                    current_node[key] = value
            else:
                current_node[key] = value

    # In case the file does not end with a separator for the last block.
    if current_node and current_node.get('Node name') != 'output':
        nodes.append(current_node)

    return nodes

def profile_activation_memory(nodes):
    """
    Create a list 'feature_maps' with length equal to the number of nodes.
    For each node whose 'Node Output Type' is 'NodeType.ACT', 
    increment the entries in feature_maps from index 'First use of this activation'
    to 'Last use of this activation' (both inclusive) by the node's 'Node memory consumption' value.
    """
    # Initialize feature_maps as a list of zeros.
    feature_maps = [0] * len(nodes)

    for node in nodes:
        if node.get("Node Output Type") == "NodeType.ACT":
            # Make sure the necessary keys are present.
            if ("First use of this activation" in node and 
                "Last use of this activation" in node and
                "Node memory consumption" in node):
                first_use = node["First use of this activation"]
                last_use = node["Last use of this activation"]
                mem_consumption = node["Node memory consumption"]
                # Ensure the indices are within bounds.
                first_use = max(0, first_use)
                last_use = min(len(feature_maps) - 1, last_use)
                # Increment all entries in the specified range.
                for i in range(first_use, last_use + 1):
                    feature_maps[i] += mem_consumption
    return feature_maps

def profile_gradient_memory(nodes):
    """
    Create a list 'gradients' with length equal to the number of nodes.
    For each node whose 'Node Output Type' is 'NodeType.GRAD', 
    increment the entries in gradients from index 'First use of this activation'
    to 'Last use of this activation' (both inclusive) by the node's 'Node memory consumption' value.
    """
    # Initialize gradients as a list of zeros.
    gradients = [0] * len(nodes)

    for node in nodes:
        if node.get("Node Output Type") == "NodeType.GRAD":
            # Make sure the necessary keys are present.
            if ("First use of this activation" in node and
                "Last use of this activation" in node and
                "Node memory consumption" in node):
                first_use = node["First use of this activation"]
                last_use = node["Last use of this activation"]
                mem_consumption = node["Node memory consumption"]
                # Ensure the indices are within bounds.
                first_use = max(0, first_use)
                last_use = min(len(gradients) - 1, last_use)
                # Increment all entries in the specified range.
                for i in range(first_use, last_use + 1):
                    gradients[i] += mem_consumption
    return gradients

if __name__ == '__main__':
    filename = 'output.txt'  # Update the path if necessary.
    node_list = parse_graph_file(filename)
    # for node in node_list:
        # print(node)

    feature_maps = profile_activation_memory(node_list)
    print("\nFeature Maps:")
    print(feature_maps)

    gradients = profile_gradient_memory(node_list)
    print("\nGradients:")
    print(gradients)
