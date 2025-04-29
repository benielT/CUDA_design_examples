import matplotlib.pyplot as plt
import networkx as nx

# Create a graph to illustrate hierarchical reduction
G = nx.DiGraph()

# Assume a block with 8 threads for simplicity
threads = [f"T{i}" for i in range(8)]
intermediates = [f"S{i}" for i in range(4)] + [f"S{i}" for i in range(4, 6)] 
final = "Global"

# Add thread nodes
G.add_nodes_from(threads + intermediates + [final])

# Assign integer values to nodes
node_values = {f"T{i}": val for i,val in enumerate([3, 1, 6, 2, 5, 4, 0, 7],0)}
node_values.update({f"S{i}": val for i,val in enumerate([4,8,9,7,12,16],0)})
node_values["Global"] = 28

# Add node attributes
nx.set_node_attributes(G, node_values, "value")

# Add edges for shared memory reduction (binary tree)
G.add_edges_from([
    ("T0", "S0"), ("T1", "S0"),
    ("T2", "S1"), ("T3", "S1"),
    ("T4", "S2"), ("T5", "S2"),
    ("T6", "S3"), ("T7", "S3"),
    ("S0", "S4"), ("S1", "S4"),
    ("S2", "S5"), ("S3", "S5"),
    ("S4", "Global"), ("S5", "Global")
])

pos = {
    "T0": (0, 0.75), "T1": (1, 0.75), "T2": (2, 0.75), "T3": (3, 0.75),
    "T4": (4, 0.75), "T5": (5, 0.75), "T6": (6, 0.75), "T7": (7, 0.75),
    "S0": (0.5, 0.5), "S1": (2.5, 0.5), "S2": (4.5, 0.5), "S3": (6.5, 0.5),
    "S4": (1.5, 0.25), "S5": (5.5, 0.25),
    "Global": (3.5, 0)
}

# Plot the graph with values as labels
plt.figure(figsize=(10, 4))
labels = nx.get_node_attributes(G, "value")
nx.draw(G, pos, with_labels=True, labels=labels, node_color='skyblue', node_size=1200, font_size=10, arrows=True)
plt.savefig("reduction_graph.png")
plt.title("Hierarchical CUDA Reduction Graph")
# plt.show
