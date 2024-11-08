#COMP-SCI 5592
#Assignment2
#Problem1

#1. Find out the best data-structure to represent / store the graph in memory.
"""
Lob(n, p) graph structure, where each star graph S_p has its center connected
to each vertex in the path graph P_n.
"""
#2.Devise an algorithm to assign the labels to the vertices using vertex k-labeling definition.(Main Task)
"""
Given a "Homogeneous Lobster+1" graph where each vertex in the path P_n connects to two star graphs S_p.
Start with an empty dictionary to store the labels for each vertex.
Assign consecutive labels to the vertices in the path P_n.
For example, assign label 1 to the first path node P_n0, and increment the label by 1 for each subsequent node.
This ensures each vertex in the path graph has a unique label.
For each node in the path P_n, two star graph centers are connected to it
Assign distinct labels to these star centers. You can alternate labels to avoid conflicts with path nodes.
For each star graph center, assign unique labels to its leaves relative to the center.
For example, if the center of S1 is labeled 2, its leaves can be labeled 4, 5, etc., to its leaves.
Repeat this for each star graph, ensuring that the leaves of the same star graph have unique labels distinct from the
center and adjacent nodes.
After the initial assignment, verify that all connected nodes have distinct labels.
If any conflicts arise (e.g., two adjacent nodes with the same label), increment the label for one of the nodes to
resolve the conflict.
"""
#3. What design strategy you will apply, also give justifications that selected strategy is most appropriate.
"""
Greedy Algorithm
Vertex labeling is inherently a sequential decision problem: we label one vertex at a time, aiming to minimize
conflicts with adjacent vertices. The Greedy approach is ideal here, as it allows us to label each vertex in a
single pass, without needing to backtrack or re-evaluate previous assignments.
The main goal of k-labeling is to ensure that no two connected vertices share the same label. By assigning labels 
incrementally and checking adjacent nodes for conflicts, the Greedy strategy can immediately rule out conflicting 
labels and proceed with the next optimal choice.
Greedy algorithms are generally efficient, with low time complexity. For this problem, we only need to visit each 
vertex once and check adjacent vertices, making the Greedy strategy computationally efficient. This low complexity 
is especially important if the graph has a large number of vertices.
Vertex k-labeling is a form of graph coloring, where we aim to assign labels (colors) to nodes such that adjacent 
nodes don’t share the same label. Greedy algorithms are widely used in graph coloring problems because they offer 
quick, approximate solutions by making locally optimal choices. This approach aligns well with the requirements of 
the k-labeling problem.
"""
#4. How traversing will be applied?
"""
Depth-First Search (DFS):
Start at a Path Node (Pn): Since the path graph Pn forms the backbone of the Lobster structure, we can start the 
traversal from any path node (e.g.,Pn0).
Traverse Through the Path Nodes: Move sequentially along the path nodes in Pn, one by one.
Visit Star Centers and Leaves: When reaching a path node, visit its connected star centers, and then recursively 
visit each leaf connected to the centers.
Backtrack to Continue Traversal: Once all nodes connected to a given path node are visited, backtrack to the next 
unvisited node along the path and repeat until all nodes in the graph are visited.
Breadth-First Search (BFS):
Initialize the Queue with a Path Node: Choose a starting path node, such as Pn0, and add it to the queue.
Dequeue a path node and visit its neighbors: first the connected star centers and then the leaves attached to those centers.
Add each unvisited neighbor to the queue to ensure they are explored in the next levels of traversal.
Continue Traversing the Path Nodes: Once all connections of the current path node are explored, move to the next path node 
in the queue and repeat the process.
Process Until the Queue is Empty: Continue until all nodes are visited and the queue is empty.
"""
#5. Store the labels of vertices and weights of the edges as an outcome.
#6. Compare your results with mathematical property and tabulate the outcomes for comparison.
#7. Hardware resources supported until what maximum value of V,E.
#8. Compute the Time Complexity of your algorithm T(V,E) or T(n).

import psutil
import os
import pandas as pd

class LobsterGraph:
    def __init__(self, n, p):
        self.n = n  # Length of path P_n
        self.p = p  # Order of each star graph S_p
        self.graph = {}  # Dictionary to store the adjacency list
        self.vertex_labels = {}  # Dictionary to store vertex labels
        self.edge_weights = {}  # Dictionary to store edge weights
        self.label_counter = 1  # Initialize label counter
        self.build_lobster_graph()

    def add_edge(self, u, v, weight=1):
        """Add an edge to the adjacency list and store its weight."""
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.edge_weights[(u, v)] = weight
        self.edge_weights[(v, u)] = weight  # For undirected graph

    def label_vertex(self, vertex):
        """Assign a label to a vertex and increment the counter."""
        self.vertex_labels[vertex] = self.label_counter
        self.label_counter += 1

    def build_lobster_graph(self):
        """Construct the Lob(n, p) graph structure and assign labels."""
        path_nodes = [f"P_{i}" for i in range(self.n)]
        for node in path_nodes:
            self.label_vertex(node)  # Label path nodes
            center1, center2 = f"{node}_S1_center", f"{node}_S2_center"
            self.label_vertex(center1)  # Label star graph centers
            self.label_vertex(center2)

            # Connect path node with star graph centers
            self.add_edge(node, center1)
            self.add_edge(node, center2)

            # Create and label leaves for each star graph
            for i in range(1, self.p + 1):
                leaf1, leaf2 = f"{center1}_leaf{i}", f"{center2}_leaf{i}"
                self.label_vertex(leaf1)
                self.label_vertex(leaf2)
                self.add_edge(center1, leaf1)
                self.add_edge(center2, leaf2)

    def display_graph(self):
        """Display the adjacency list to show the graph structure"""
        for node, neighbors in self.graph.items():
            print(f"{node}: {neighbors}")

    def get_actual_values(self):
        """Return the actual number of vertices and edges."""
        return len(self.vertex_labels), sum(len(neighbors) for neighbors in self.graph.values()) // 2

    def get_theoretical_values(self):
        """Calculate theoretical values of vertices and edges."""
        theoretical_V = self.n * (2 * self.p + 3)
        theoretical_E = self.n * (2 * self.p + 1) - 1
        return theoretical_V, theoretical_E

    def get_labels_and_weights(self):
        """Return vertex labels and edge weights."""
        return self.vertex_labels, self.edge_weights

    def compute_time_complexity(self):
        """Compute the time complexity based on the actual number of vertices and edges."""
        V, E = self.get_actual_values()
        return f"O({V} + {E})"

class VertexLabeling:
    def __init__(self, n, p):
        self.n = n  # Number of nodes in the path graph P_n
        self.p = p  # Order of each star graph S_p
        self.labels = {}  # Dictionary to store the labels for each vertex
        self.current_label = 1  # Start labeling from label 1

    def assign_labels(self):
        # Label path nodes Pn
        path_nodes = [f"P_{i}" for i in range(self.n)]
        for node in path_nodes:
            self.labels[node] = self.current_label
            self.current_label += 1

        # Label star centers and leaves for each path node
        for node in path_nodes:
            # Assign labels to star centers connected to this path node
            center1, center2 = f"{node}_S1_center", f"{node}_S2_center"
            self.labels[center1] = self.current_label
            self.current_label += 1
            self.labels[center2] = self.current_label
            self.current_label += 1

            # Assign labels to the leaves of each star graph
            for i in range(1, self.p + 1):
                leaf1, leaf2 = f"{center1}_leaf{i}", f"{center2}_leaf{i}"
                self.labels[leaf1] = self.current_label
                self.current_label += 1
                self.labels[leaf2] = self.current_label
                self.current_label += 1

    def display_labels(self):
        for node, label in self.labels.items():
            print(f"{node}: {label}")

class LobsterGraphTraversal:
    def __init__(self, lobster_graph):
        self.graph = lobster_graph.graph

    def dfs(self, start):
        """Recursive DFS traversal starting from a node."""
        visited = set()
        self._dfs_helper(start, visited)

    def _dfs_helper(self, node, visited):
        """Recursive helper function for DFS."""
        if node not in visited:
            print(node)
            visited.add(node)
            for neighbor in self.graph.get(node, []):
                self._dfs_helper(neighbor, visited)

    def bfs(self, start):
        """Perform BFS traversal starting from a node."""
        visited = set([start])
        queue = [start]
        while queue:
            node = queue.pop(0)
            print(node)
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

class LobsterGraphTester:
    def __init__(self, memory_limit_mb=16000):
        self.max_vertices = 0
        self.max_edges = 0
        self.memory_limit_mb = memory_limit_mb
        self.progress_threshold = 5  # Start progress display at 5%

    def get_process_memory(self):
        """Get current Python process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

    def test_max_lobster_graph(self, initial_n=1, initial_p=1, increment=1):
        """Gradually increase n and p, checking memory usage to find max supported vertices and edges."""
        n, p = initial_n, initial_p
        while True:
            try:
                graph = LobsterGraph(n, p)
                V, E = graph.get_actual_values()
                memory_used_mb = self.get_process_memory()
                memory_used_percent = (memory_used_mb / self.memory_limit_mb) * 100

                if memory_used_percent >= self.progress_threshold:
                    print(f"Progress: {int(memory_used_percent)}% - Current n: {n}, p: {p} - "
                          f"Vertices: {V}, Edges: {E}, Process memory usage: {memory_used_mb:.2f} MB")
                    self.progress_threshold += 5

                if memory_used_mb > self.memory_limit_mb:
                    print("Memory limit reached. Stopping...")
                    break

                self.max_vertices = V
                self.max_edges = E
                n += increment
                p += increment

            except MemoryError:
                print("MemoryError: Maximum memory capacity reached.")
                break

    def get_max_supported_values(self):
        """Return the maximum supported values for vertices and edges."""
        return self.max_vertices, self.max_edges

n, p = 3, 3
lobster_graph = LobsterGraph(n, p)
print("Adjacency List of the Graph:")
lobster_graph.display_graph()

vertex_labeling = VertexLabeling(n, p)
vertex_labeling.assign_labels()
vertex_labeling.display_labels()

lobster_traversal = LobsterGraphTraversal(lobster_graph)
print("\nDFS Traversal:")
lobster_traversal.dfs("P_0")
print("\nBFS Traversal:")
lobster_traversal.bfs("P_0")

tester = LobsterGraphTester(memory_limit_mb=16000)
tester.test_max_lobster_graph()
max_V, max_E = tester.get_max_supported_values()
print("Max supported vertices (V):", max_V)
print("Max supported edges (E):", max_E)

results = []
for n, p in [(3, 3), (3, 5), (5, 3), (5, 5), (10, 3), (10, 5)]:
    graph = LobsterGraph(n, p)
    actual_V, actual_E = graph.get_actual_values()
    theoretical_V, theoretical_E = graph.get_theoretical_values()
    time_complexity = graph.compute_time_complexity()
    vertex_labels, edge_weights = graph.get_labels_and_weights()
    results.append({
        "n": n,
        "p": p,
        "Calculated V": theoretical_V,
        "Actual V": actual_V,
        "Calculated E": theoretical_E,
        "Actual E": actual_E,
        "Time Complexity": time_complexity,
        "Vertex Labels": vertex_labels,
        "Edge Weights": edge_weights
    })

# Displaying the results in a tabular format using pandas
df = pd.DataFrame(results)
print("Comparison Table:")
print(df[["n", "p", "Calculated V", "Actual V", "Calculated E", "Actual E", "Time Complexity"]])

# Save as an Excel file
df.to_excel("Problem1_Results.xlsx", index=False)
print("Results saved to Problem1_Results.xlsx")

# Output vertex labels and edge weights for each graph
for index, row in df.iterrows():
    print(f"\nLobsterGraph with n={row['n']} and p={row['p']}")
    print("Vertex Labels:")
    print(row["Vertex Labels"])
    print("Edge Weights:")
    print(row["Edge Weights"])
    print("Time Complexity:", row["Time Complexity"])

