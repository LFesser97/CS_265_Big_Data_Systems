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
            elif key == 'Input to this node':
                items = value.strip("[]").split(",")
                underlying_list = [item.strip() for item in items]
                current_node[key] = underlying_list
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

def compute_activation_inactive_times(nodes):
    """
    For each activation node, compute the 'idle time' defined as the sum of
    execution times of nodes between its last usage in the forward pass (i.e.
    before the 'sep' token) and its first usage in the backward pass (after the 'sep' token).
    
    Only nodes that actually list the activation as an input (via the "Input to this node"
    field) are considered as valid usage points.
    
    Returns a dictionary where keys are the activation node names and values are the idle times.
    """
    # Find the index of the 'sep' token in the nodes list.
    sep_index = None
    for idx, node in enumerate(nodes):
        if node.get("Node name") == "sep":
            sep_index = idx
            break
    if sep_index is None:
        raise ValueError("Separator token 'sep' not found in the nodes list.")
    
    idle_times = {}
    
    # Loop over each node to process activation nodes.
    for node in nodes:
        if node.get("Node Output Type") == "NodeType.ACT":
            # Ensure the necessary usage keys exist.
            if ("First use of this activation" in node and 
                "Last use of this activation" in node):
                first_use = node["First use of this activation"]
                last_use = node["Last use of this activation"]
                activation_name = node.get("Node name")
                
                # Identify valid forward usage indices where the activation is used as input.
                forward_usage = [
                    i for i in range(first_use, last_use + 1)
                    if i < sep_index and activation_name in nodes[i].get("Input to this node", "")
                ]
                # Identify valid backward usage indices where the activation is used as input.
                backward_usage = [
                    i for i in range(first_use, last_use + 1)
                    if i > sep_index and activation_name in nodes[i].get("Input to this node", "")
                ]
                
                # If either list is empty, no valid usage was found.
                if not forward_usage or not backward_usage:
                    idle_times[activation_name] = 0
                    continue
                
                forward_last = max(forward_usage)
                backward_first = min(backward_usage)
                
                # Sum execution times for nodes between forward_last and backward_first.
                idle_time = 0
                # Sum nodes with indices from forward_last+1 to backward_first-1.
                for i in range(forward_last + 1, backward_first):
                    if "Node execution time" in nodes[i]:
                        idle_time += nodes[i]["Node execution time"]
                
                idle_times[activation_name] = idle_time
            else:
                # If usage keys are missing, set idle time to 0.
                idle_times[node.get("Node name")] = 0
    
    return idle_times

def compute_activation_recompute_ratio(nodes):
    """
    For each activation node (i.e. node with "Node Output Type" == "NodeType.ACT"), compute:

      - recompute_time: the sum of "Node execution time" for all nodes that occur
        earlier in the computation graph (i.e. at lower indices).
      - recompute ratio: defined as (Node memory consumption) / (recompute_time).

    The result is returned as a dictionary where keys are the activation node names
    and values are the corresponding recompute ratios.

    If the recompute_time is zero, the ratio is set to 0 to avoid division by zero.
    """
    recompute_ratios = {}

    # Iterate over nodes along with their indices.
    for idx, node in enumerate(nodes):
        if node.get("Node Output Type") == "NodeType.ACT":
            # Ensure the node has the necessary 'Node memory consumption' key.
            if "Node memory consumption" not in node:
                continue  # or you can assign a default value if needed.
            mem_consumption = node["Node memory consumption"]

            # Compute the recompute time: sum execution times of all nodes before this one.
            recompute_time = 0
            for j in range(idx):
                if "Node execution time" in nodes[j]:
                    recompute_time += nodes[j]["Node execution time"]

            # Protect against division by zero.
            if recompute_time == 0:
                ratio = 0
            else:
                ratio = mem_consumption / recompute_time
                # ratio = recompute_time

            # Store the ratio in the dictionary, keyed by the node's name.
            activation_name = node.get("Node name", f"activation_{idx}")
            recompute_ratios[activation_name] = ratio

    return recompute_ratios

if __name__ == '__main__':
    filename = 'output.txt'  # Update the path if necessary.
    node_list = parse_graph_file(filename)
    # for node in node_list:
        # print(node)

    feature_maps = profile_activation_memory(node_list)
    # print("\nFeature Maps:")
    # print(feature_maps)

    gradients = profile_gradient_memory(node_list)
    # print("\nGradients:")
    # print(gradients)

    idle_times = compute_activation_inactive_times(node_list)
    sorted_idle_times = {k: v for k, v in sorted(idle_times.items(), key=lambda item: item[1], reverse=True)}
    # print(sorted_idle_times)

    recompute_ratios = compute_activation_recompute_ratio(node_list)
    sorted_recompute_ratios = {k: v for k, v in sorted(recompute_ratios.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_recompute_ratios)
