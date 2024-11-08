#COMP-SCI 5592
#Assignment2
#Problem4

# 1. Suggest a suitable name.
"""
Centroid-Star Graph
"""

# 2. Formulate the equations to calculate the order and size of the graph.
"""
Order (Number of Vertices): If the graph has one central vertex connected to multiple branches, 
and each branch has n vertices, then the total number of vertices V can be expressed as: 
V = 1 + n * k
Size (Number of Edges): Each branch has n edges extending from the center. 
For k branches, the total number of edges E is:
E = n * k
"""

# 3. Data structure to store the graph.
# 4. Use an algorithm to assign labels.
# 5. Store the labels of vertices and weights of edges as a result.

# Implementation of the adjacency list data structure
class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_vertex(self, vertex):
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, u, v, weight=1):
        self.adj_list[u].append((v, weight))
        self.adj_list[v].append((u, weight))  # For undirected graph, add edges in both directions

    def display(self):
        for vertex, edges in self.adj_list.items():
            print(f"Vertex {vertex}:", end=" ")
            for edge in edges:
                print(f" -> {edge[0]} (Weight {edge[1]})", end=" ")
            print()

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            for vertex, edges in self.adj_list.items():
                file.write(f"Vertex {vertex}:")
                for edge in edges:
                    file.write(f" -> {edge[0]} (Weight {edge[1]})")
                file.write("\n")


def create_centroid_star_graph(k, n):
    graph = Graph()

    # Add the central vertex
    graph.add_vertex(0)

    label_counter = 1
    for branch in range(k):
        branch_start_node = label_counter
        graph.add_vertex(branch_start_node)
        graph.add_edge(0, branch_start_node, weight=branch + 1)  # Set edge weight as branch index + 1
        label_counter += 1

        # Add other nodes in the branch
        for node in range(1, n):
            graph.add_vertex(branch_start_node + node)
            graph.add_edge(branch_start_node + node - 1, branch_start_node + node, weight=node)
            label_counter += 1

    return graph


k = 4  # Number of branches
n = 5  # Number of nodes per branch

# Create the graph
graph = create_centroid_star_graph(k, n)

# Display graph information to the console
graph.display()

# Save graph information to a file
output_filename = "Problem4_Results.txt"
graph.save_to_file(output_filename)
print(f"Graph information has been saved to file {output_filename}")
