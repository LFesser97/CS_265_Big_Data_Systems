from enum import Enum
from typing import Dict, Any, Set, Tuple, List
import torch
import torch.fx as fx

class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"

class NodeType(Enum):
    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3

class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        optimizer_node = None
        for node in self.module.graph.nodes:
            if node.op == OP.CALL_FUNCTION and node.target == torch.ops.aten._fused_adam.default:
                optimizer_node = node
                break

        # If found, extract the parameter and gradient nodes from its arguments.
        params = set()
        grads = set()
        if optimizer_node is not None:
            # The first argument is assumed to be a list of parameter nodes.
            params = set(optimizer_node.args[0])
            # The second argument is assumed to be a list of gradient nodes.
            grads = set(optimizer_node.args[1])

        # First, perform an initial categorization.
        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                if node in params:
                    node.node_type = NodeType.PARAM
                elif node in grads:
                    node.node_type = NodeType.GRAD
                else:
                    node.node_type = NodeType.OTHER
            else:
                # For non-placeholder nodes, assume the computed result is an activation.
                node.node_type = NodeType.ACT

        # Rule 1: Any node with a name that starts with "arg0_" gets classified as PARAM.
        for node in self.module.graph.nodes:
            if isinstance(node.name, str) and node.name.startswith("arg0_"):
                node.node_type = NodeType.PARAM

        # Rule 2: Any node whose name starts with "t_" that has a PARAM input also gets classified as PARAM.
        for node in self.module.graph.nodes:
            if isinstance(node.name, str) and node.name.startswith("t"):
                # If any of the input nodes is classified as PARAM, then classify this node as PARAM.
                for input_node in node.all_input_nodes:
                    if getattr(input_node, "node_type", None) == NodeType.PARAM:
                        node.node_type = NodeType.PARAM
                        break

        # Rule 3: Any node that appears after the node named "sep" gets classified as GRAD.
        sep_found = False
        for node in self.module.graph.nodes:
            if node.name == "sep":
                sep_found = True
                continue  # We leave the "sep" node's own type as is.
            if sep_found:
                node.node_type = NodeType.GRAD

        # static analysis to determine first and last usage of each activation
        self.run_static_analysis(self.module)
        
        # subgraph extraction for later recomputation
        subgraph, users = self.extract_subgraph(self.module, "convolution")
        print("Upstream subgraph nodes:", {node.name for node in subgraph.graph.nodes})
        print("Users of this node:", users)
        
        # adjusted_node_list = self.insert_subgraph(subgraph, "convolution", "cudnn_batch_norm_backward_19", users)
        # print("Adjusted graph nodes:", adjusted_node_list)
        self.module = self.insert_subgraph(subgraph, "convolution", "cudnn_batch_norm_backward_19", users)

        """
        # Print details about each node.
        for node in self.module.graph.nodes:
            print("Node name:", node.name)
            print("Node op:", node.op)
            print("Node target:", node.target)
            print("Input to this node:", node.all_input_nodes)
            print("Users of this node:", node.users)
            print("Node Output Type", getattr(node, "node_type", "Not categorized"))
            if getattr(node, "node_type", None) == NodeType.ACT:
                print("First use of this activation:", getattr(node, "first_appearance", "First appearance unknown"))
                print("Last use of this activation:", getattr(node, "last_appearance", "Last appearance unknown"))
            print("Node memory consumption:", getattr(node, "mem_used", "Not profiled"))
            print("Node execution time:", getattr(node, "exec_time", "Not profiled"))
            print("-" * 40)
        """

    def run(self,
            *args,
            initial_env: Dict[fx.Node, Any] | None = None,
            enable_io_processing: bool = True) -> Any:
        return super().run(*args, initial_env=initial_env, enable_io_processing=enable_io_processing)

    def run_node(self, n: fx.Node) -> Any:
        if torch.cuda.is_available():
            # Record memory usage before executing the node.
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            print("WARNING. No GPU found.")
            mem_before = 0

        # Execute the node normally.
        result = super().run_node(n)

        if torch.cuda.is_available():
            end_event.record()
            # Wait for the events to be recorded.
            torch.cuda.synchronize()
            exec_time = start_event.elapsed_time(end_event)  # milliseconds
            mem_after = torch.cuda.memory_allocated()
            mem_used = mem_after - mem_before
        else:
            exec_time = 0.0
            mem_used = 0

        n.mem_used = mem_used
        n.exec_time = exec_time
       
        # Print details about the current node
        print("Node name:", n.name)
        print("Node op:", n.op)
        print("Node target:", n.target)
        print("Input to this node:", n.all_input_nodes)
        print("Users of this node:", n.users)
        print("Node Output Type:", getattr(n, "node_type", "Not categorized"))
        if getattr(n, "node_type", None) == NodeType.ACT or getattr(n, "node_type", None) == NodeType.GRAD:
            print("First use of this activation:", getattr(n, "first_appearance", "First appearance unknown"))
            print("Last use of this activation:", getattr(n, "last_appearance", "Last appearance unknown"))
        print("Node memory consumption:", getattr(n, "mem_used", "Not profiled"))
        print("Node execution time:", getattr(n, "exec_time", "Not profiled"))
        print("-" * 40)
        
        return result
        

    def run_static_analysis(self, graph_module: fx.GraphModule) -> None:
        node_indices = {node: idx for idx, node in enumerate(graph_module.graph.nodes)}
        # Iterate over the graph nodes in order (the order reflects their appearance).
        for idx, node in enumerate(graph_module.graph.nodes):
            # Check if the node is classified as an activation.
            if getattr(node, "node_type", None) == NodeType.ACT or getattr(node, "node_type", None) == NodeType.GRAD:
                # Only add the attribute if it hasn't been set already.
                if not hasattr(node, "first_appearance"):
                    node.first_appearance = idx
                    
                # If the node is used by any other node, determine the last appearance.
                if node.users:
                    # Use the node_indices mapping to get the index of each user.
                    last_idx = max(node_indices[user] for user in node.users if user in node_indices)
                else:
                    # If no users exist, the last appearance is the same as the first.
                    last_idx = idx

                node.last_appearance = last_idx

    def aggregate_stats(self) -> None:
        pass

    def print_stats(self) -> None:
        pass

    def reset_stats(self) -> None:
        pass
    
    def extract_subgraph(self,
                         graph_module: fx.GraphModule,
                         target_node_name: str
                        ) -> Tuple[fx.GraphModule, List[str]]:
        """
        Extracts the upstream subgraph for a given target node in an FX GraphModule while excluding
        any nodes whose name starts with 'arg'. The extracted subgraph includes all nodes that produce 
        values consumed by the target node. Instead of modifying the original nodes, the function
        creates a duplicate subgraph where each node is copied and its name is prefixed with 're_' (if not already).
        Additionally, it returns a list of names of the target node's direct user nodes.

        Args:
            graph_module (fx.GraphModule): The FX GraphModule containing the original computation graph.
            target_node_name (str): The name of the target node marking the end of the subgraph.

        Returns:
            Tuple[fx.GraphModule, List[str]]:
                - A new FX GraphModule representing the extracted (duplicated) subgraph.
                - A list containing the names of the target node's user nodes.

        Raises:
            ValueError: If no node with the specified target name is found in the graph.
        """
        # 1. Locate the target node.
        target_node = None
        for node in graph_module.graph.nodes:
            if node.name == target_node_name:
                target_node = node
                break
        if target_node is None:
            raise ValueError(f"Node with name '{target_node_name}' not found in the graph.")

        # 2. Recursively collect all upstream nodes (filter out nodes whose name starts with 'arg').
        collected: List[fx.Node] = []
        def _collect_upstream(node: fx.Node, collected: List[fx.Node]) -> None:
            if isinstance(node.name, str) and node.name.startswith("arg"):
                return
            if node in collected:
                return
            collected.append(node)
            # Traverse through positional arguments.
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    _collect_upstream(arg, collected)
            # Traverse through keyword arguments.
            for value in node.kwargs.values():
                if isinstance(value, fx.Node):
                    _collect_upstream(value, collected)

        _collect_upstream(target_node, collected)

        # 3. Sort the collected nodes to preserve their original topological order.
        original_nodes = list(graph_module.graph.nodes)
        extracted_nodes = sorted(collected, key=lambda n: original_nodes.index(n))

        # 4. Instead of renaming the original nodes, build a mapping of new names.
        #    For nodes that do not already start with 're_', the new name will be 're_' + original_name.
        new_names = {}
        for node in extracted_nodes:
            if isinstance(node.name, str) and not node.name.startswith("re_"):
                new_names[node] = "re_" + node.name
            else:
                new_names[node] = node.name

        # 5. Build a list of the target node's direct user names.
        user_names = [user.name for user in target_node.users]

        # 6. Create a new FX graph and duplicate the extracted nodes into it using the new names.
        new_graph = fx.Graph()
        node_mapping = {}
        for node in extracted_nodes:
            # Re-map arguments: if an argument is a node already copied, substitute it.
            new_args = tuple(node_mapping.get(arg, arg) if isinstance(arg, fx.Node) else arg for arg in node.args)
            new_kwargs = {k: node_mapping.get(v, v) if isinstance(v, fx.Node) else v for k, v in node.kwargs.items()}
            new_node = new_graph.create_node(node.op, node.target, new_args, new_kwargs, name=new_names[node])
            node_mapping[node] = new_node

        # 7. Set the output of the new graph to be the duplicate of the target node.
        new_graph.output(node_mapping[target_node])

        # 8. Create a new FX GraphModule from the new graph using self.module as the root module.
        new_subgraph_gm = fx.GraphModule(self.module, new_graph)

        return new_subgraph_gm, user_names
    
    
    def insert_subgraph(self,
                        graph_module: fx.GraphModule,
                        target_name: str,
                        earliest_bw_use: str,
                        users: List
                       ) -> fx.GraphModule:
        # Get the nodes from the extracted subgraph.
        # These nodes should already be ordered in a valid topological order.
        extracted_nodes = [node for node in list(graph_module.graph.nodes) if node.name != "output"]

        # Access the current computation graph via self.module.
        current_graph = self.module.graph
        current_nodes = list(current_graph.nodes)

        # Find the insertion index where the node name matches earliest_bw_use.
        insertion_index = None
        for i, node in enumerate(current_nodes):
            if node.name == earliest_bw_use:
                insertion_index = i
                break
        if insertion_index is None:
            raise ValueError(f"Node with name '{earliest_bw_use}' not found in the current graph.")
            
        new_target_name = "re_" + target_name
        for node in current_nodes[insertion_index:]:
            if node.name in users:
                for input_node in node.all_input_nodes:
                    if input_node.name == target_name:
                        input_node.name = new_target_name

        # Create a new node ordering by inserting the extracted subgraph nodes 
        # right before the found insertion index.
        new_node_list = current_nodes[:insertion_index] + extracted_nodes + current_nodes[insertion_index:]

        # Update the current graph's internal node list.
        current_graph._nodes = new_node_list

        # Recompile self.module to update its forward method and reflect graph changes.
        self.module.recompile()

        return self.module
        # return new_node_list