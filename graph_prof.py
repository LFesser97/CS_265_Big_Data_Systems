from enum import Enum
from typing import Dict
import torch
import torch.fx as fx
from typing import Dict, Any


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # Create a dictionary to hold profiling stats per node.
        self.stats: Dict[str, Dict[str, Any]] = {}

        # Build a set of IDs for all module parameters to classify inputs/outputs.
        self.param_ids = {id(p) for p in self.module.parameters()}

        nodes_list = list(self.module.graph.nodes)
        # Printing the input nodes, node users and node names.
        for node in nodes_list:
            print("Node name: ", node.name)
            print("Node type: ", node.op)
            print("Node target: ", node.target)
            print("Input to this node", node.all_input_nodes)
            print("Users of this node: ", node.users)

        # Build an index mapping from node to its order in the graph.
        self.node_index = {node: idx for idx, node in enumerate(nodes_list)}

        # Identify the forward/backward boundary.
        # We assume the first node with target torch.ops.separator.sep.default marks the end of the forward pass.
        self.sep_index = None
        for node in nodes_list:
            # Adjust the target check based on how the separator is defined.
            if node.op == "call_function" and getattr(node.target, '__name__', None) == "sep":
                self.sep_index = self.node_index[node]
                break
        # If no separator is found, treat the entire graph as forward.
        if self.sep_index is None:
            self.sep_index = len(nodes_list)

        # Analyze static activation usage for nodes that produce activations.
        # We consider nodes that are not placeholders, get_attr, or output as candidates.
        self.activation_usage = {}
        for node in nodes_list:
            if node.op in ("placeholder", "get_attr", "output"):
                continue
            # We only care if the node has users.
            if not node.users:
                continue
            # For each user, separate those in the forward pass (index < sep_index)
            # and those in the backward pass (index >= sep_index).
            forward_users = []
            backward_users = []
            for user in node.users:
                idx = self.node_index.get(user)
                if idx is None:
                    continue
                if idx < self.sep_index:
                    forward_users.append(idx)
                else:
                    backward_users.append(idx)
            first_use_forward = min(forward_users) if forward_users else None
            last_use_forward = max(forward_users) if forward_users else None
            first_use_backward = min(backward_users) if backward_users else None
            last_use_backward = max(backward_users) if backward_users else None

            self.activation_usage[node.name] = {
                "first_use_forward": first_use_forward,
                "last_use_forward": last_use_forward,
                "first_use_backward": first_use_backward,
                "last_use_backward": last_use_backward,
            }
            # If this node already has stats (e.g. from a run), add the activation usage.
            if node.name in self.stats:
                self.stats[node.name]["activation_usage"] = self.activation_usage[node.name]
            else:
                # Otherwise, initialize the stats with only activation usage.
                self.stats[node.name] = {"activation_usage": self.activation_usage[node.name]}

    def categorize_value(self, value: Any) -> str:
        # Categorize a value as one of: parameter, gradient, activation, optimizer state, or other.
        # If the value is a tensor, check if it is one of the module parameters.
        if isinstance(value, torch.Tensor):
            if id(value) in self.param_ids:
                return "parameter"
            # Heuristic: if the tensor requires grad and has a non-None grad, mark it as gradient.
            # (Note: Depending on your use case, you might refine this logic.)
            elif value.requires_grad and value.grad is not None:
                return "gradient"
            else:
                return "activation"
        # If the value is a list or tuple, we might be looking at a collection of states (e.g., optimizer state).
        elif isinstance(value, (list, tuple)):
            # Check if at least one element is a tensor and whether it represents a parameter.
            if any(isinstance(v, torch.Tensor) and id(v) in self.param_ids for v in value):
                return "parameter"
            else:
                return "optimizer state"
        return "other"

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> Any:
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

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

        # Retrieve input values from the environment.
        # Note: n.all_input_nodes gives nodes that provided inputs.
        input_values = [self.env[node] for node in n.all_input_nodes if node in self.env]
        input_types = [self.categorize_value(val) for val in input_values]
        output_type = self.categorize_value(result)

        # Record stats for the node.
        if n.name not in self.stats:
            self.stats[n.name] = {
                "op": n.op,
                "target": n.target,
                "times": [],
                "memories": [],
                "input_types": [],  # list of lists, one per run
                "output_types": []  # list of output type strings per run
            }
            # If we already computed static activation usage for this node, add it.
            if n.name in self.activation_usage:
                self.stats[n.name]["activation_usage"] = self.activation_usage[n.name]
       
        else:
            # If the node already exists, ensure the required keys are present.
            for key in ["times", "memories", "input_types", "output_types"]:
                self.stats[n.name].setdefault(key, [])

        self.stats[n.name]["times"].append(exec_time)
        self.stats[n.name]["memories"].append(mem_used)
        self.stats[n.name]["input_types"].append(input_types)
        self.stats[n.name]["output_types"].append(output_type)

        # If in the backward pass and a feature map was swapped out, we can swap it back here.
        # Also, in the forward pass, we can swap out memory for intermediate results if needed.

        return result

    def aggregate_stats(self) -> None:
        # You are expected run the profiler for x warm-up iterations and y
        # actual measurement iterations. The run-time measurement then needs to
        # be averaged over the y runs.
        for node_name, data in self.stats.items():
            if data["times"]:
                avg_time = sum(data["times"]) / len(data["times"])
            else:
                avg_time = 0.0

            if data["memories"]:
                avg_mem = sum(data["memories"]) / len(data["memories"])
            else:
                avg_mem = 0

            data["avg_time"] = avg_time
            data["avg_mem"] = avg_mem

    def print_stats(self) -> None:
        # Print the averaged computation time and memory usage for each operator.
        print("Operator profiling stats:")
        for node_name, data in self.stats.items():
            avg_time = data.get("avg_time", "N/A")
            avg_mem = data.get("avg_mem", "N/A")
            print(f"Node: {node_name}, op: {data['op']}, target: {data['target']}")
            print(f"  Average execution time (ms): {avg_time}")
            print(f"  Average memory change (bytes): {avg_mem}")
            # Print the categories of the inputs and outputs for the first run as an example.
            if data["input_types"]:
                print(f"  Example input types: {data['input_types'][0]}")
            if data["output_types"]:
                print(f"  Example output type: {data['output_types'][0]}")
            # If activation usage information is available, print it.
            if "activation_usage" in data:
                au = data["activation_usage"]
                print("  Activation usage:")
                print(f"    First use in forward pass at node index: {au.get('first_use_forward')}")
                print(f"    Last use in forward pass at node index: {au.get('last_use_forward')}")
                print(f"    First use in backward pass at node index: {au.get('first_use_backward')}")
                print(f"    Last use in backward pass at node index: {au.get('last_use_backward')}")

    def reset_stats(self) -> None:
        # The statistics must be cleared out after x warm-up iterations and
        # reset before the actual measurement begins.
        self.stats = {}
