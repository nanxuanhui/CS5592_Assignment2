#COMP-SCI 5592
#Assignment2
#Problem3

# 1. Suggest a suitable name.
"""
Radial Branch Graph
"""
# 2. Devise the formulae for calculating order and size of the graph.
"""
Order (number of vertices, V): Let n represent the number of branches, and each branch consists of a series of vertices 
connected to the centroid. If each branch has a set number of vertices k, then the order of the graph can be calculated as:
V = 1 + n * k
Size (number of edges, E): If each branch connects sequentially to the centroid or other vertices within the branch, 
the size of the graph can be:
E = n * (k + 1) - n
"""
# 3. Data structure to store the graph.
# 4. Assign the labels using algorithm.
# 5. Store the labels of vertices and weights of the edges as an outcome.

# Define the function to build the graph
def build_graph(n_branches, nodes_per_branch):
    # Initialize the adjacency list and label dictionary
    graph = {}
    graph_labels = {}
    edge_weights = {}

    # Define the centroid node and assign it the label 0
    graph["centroid"] = []
    graph_labels["centroid"] = 0

    label_counter = 1  # Start assigning labels from 1 for other nodes

    # Create each branch
    for branch in range(1, n_branches + 1):
        previous_node = "centroid"  # Each branch starts from the centroid

        # Create a name and assign a label for each node in the branch
        for node in range(1, nodes_per_branch + 1):
            node_name = f"branch{branch}_node{node}"
            graph[node_name] = []
            graph_labels[node_name] = label_counter  # Assign label
            label_counter += 1

            # Add edge and store the edge weight
            graph[previous_node].append(node_name)
            graph[node_name].append(previous_node)
            edge_weights[(previous_node, node_name)] = 1
            edge_weights[(node_name, previous_node)] = 1

            # Update the previous node to the current node so the next node connects to the current one
            previous_node = node_name

    return graph, graph_labels, edge_weights


# Function to save the graph structure to a txt file
def save_graph_to_txt(graph, graph_labels, edge_weights, filename="Problem3_Results.txt"):
    with open(filename, "w") as file:
        # Save the adjacency list
        file.write("Adjacency List:\n")
        for node, neighbors in graph.items():
            neighbors_str = ", ".join(neighbors)
            file.write(f"{node}: [{neighbors_str}]\n")

        file.write("\nVertex Labels:\n")
        for node, label in graph_labels.items():
            file.write(f"{node}: {label}\n")

        file.write("\nEdge Weights:\n")
        for edge, weight in edge_weights.items():
            file.write(f"{edge}: {weight}\n")
    print(f"Results have been saved to {filename}")


# Set parameters: n_branches is the number of branches, nodes_per_branch is the number of nodes per branch
n_branches = 5  # Assume there are 5 branches
nodes_per_branch = 3  # Each branch has 3 nodes

# Build the graph
graph, graph_labels, edge_weights = build_graph(n_branches, nodes_per_branch)

# Save the results to a txt file
save_graph_to_txt(graph, graph_labels, edge_weights)