from enum import Enum
from typing import Dict, Any
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
        print("Graph Nodes ", self.module.graph.nodes)
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

