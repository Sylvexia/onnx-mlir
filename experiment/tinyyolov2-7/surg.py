import onnx
import onnx_graphsurgeon as gs
from collections import deque
import numpy as np

def remove_node_and_downstream(model_path, target_node_name, output_path):
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)

    # Build tensor -> consumer map
    tensor_consumers = {}
    for node in graph.nodes:
        for tensor in node.inputs:
            tensor_consumers.setdefault(tensor.name, []).append(node)

    # Find the target node
    target_node = next((n for n in graph.nodes if n.name == target_node_name), None)
    if not target_node:
        raise ValueError(f"Node '{target_node_name}' not found.")

    # BFS to find all downstream nodes (by following tensor consumers)
    to_visit = deque([target_node])
    visited_names = set()
    nodes_to_remove = []

    while to_visit:
        node = to_visit.popleft()
        if node.name in visited_names:
            continue
        visited_names.add(node.name)
        nodes_to_remove.append(node)

        for output_tensor in node.outputs:
            for consumer in tensor_consumers.get(output_tensor.name, []):
                if consumer.name not in visited_names:
                    to_visit.append(consumer)

    # Remove the nodes
    graph.nodes = [n for n in graph.nodes if n.name not in visited_names]

    # Remove outputs that are now dangling
    removed_tensor_names = {t.name for n in nodes_to_remove for t in n.outputs}
    graph.outputs = [o for o in graph.outputs if o.name not in removed_tensor_names]

    # Now it's safe to clean up
    graph.cleanup().toposort()
    # Finalize and export
    # graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_path)
    print(f"Removed {len(nodes_to_remove)} downstream nodes from '{target_node_name}'. Saved to '{output_path}'.")

# Example usage:
remove_node_and_downstream("/home/sylvex/onnx-mlir/experiment/tinyyolov2-7/Model/Model.onnx", "convolution7", "cleaned_model.onnx")