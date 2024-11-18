#COMP-SCI 5592
#Assignment2
#Problem2

#1. Find out the best data-structure to represent / store the graph in memory.
#2. Devise an algorithm to assign the labels to the vertices using vertex k-labeling definition. (Main Task)
"""
Start by assigning the label 1 to the central node.
For each branch extending from the central node, assign labels incrementally along the branch.
By incrementing labels along each branch, ensure that the weight of each edge (the sum of adjacent node labels)
is unique within each branch. To avoid duplicate edge weights across different branches, an incremental offset
can be added to the starting label of each branch based on the branch index. This approach ensures that edge
weights are unique across the entire graph. Calculate the edge weights and store them in a set to verify that
all edge weights are unique.
"""
#3. What design strategy you will apply, also give justifications that selected strategy is most appropriate.
"""
Greedy Labeling with Incremental Constraints
The greedy method is highly efficient for node labeling as it minimizes the need for complex calculations or 
backtracking. Each label assignment is based on a simple incremental rule, resulting in low computational 
requirements. Ensuring the uniqueness of edge weights is the key constraint in this problem. The greedy 
method with incremental constraints systematically prevents duplicate weights by controlling the increment 
pattern within and between branches. This strategy is simple to implement, involving only label assignment 
and weight calculation operations. For large graphs with more branches or longer branches, this method scales 
well, as it can be applied consistently regardless of the number of vertices or branches.
"""
#4. How traversing will be applied?
#5. Store the labels of vertices and weights of the edges as an outcome.
#6. Compare your results with mathematical property and tabulate the outcomes for comparison.
#7. Hardware resources supported until what maximum value of V,E.
#8. Compute the Time Complexity of your algorithm T(V,E) or T(n).

import psutil
from collections import deque

class StarGraph1:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.center = 1
        self.graph = {}
        self.edge_weights = {}
        self.build_graph()

    def build_graph(self):
        self.graph[self.center] = []
        vertex_label = 2
        for branch in range(self.m):
            previous_vertex = self.center
            for i in range(self.n):
                self.graph[vertex_label] = [previous_vertex]
                self.graph[previous_vertex].append(vertex_label)
                previous_vertex = vertex_label
                vertex_label += 1

    def assign_edge_weights(self):
        weight = 1
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                if (node, neighbor) not in self.edge_weights and (neighbor, node) not in self.edge_weights:
                    self.edge_weights[(node, neighbor)] = weight
                    weight += 1

    def display_graph(self):
        print("Adjacency List:")
        for node, neighbors in self.graph.items():
            print(f"{node}: {neighbors}")

        print("\nEdge Weights:")
        for edge, weight in self.edge_weights.items():
            print(f"Edge {edge} -> Weight: {weight}")

class StarGraph2:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.center = 1
        self.graph = {}
        self.vertex_labels = {}
        self.edge_weights = []
        self.build_graph()
        self.assign_vertex_labels()
        self.compute_edge_weights()

    def build_graph(self):
        self.graph[self.center] = []
        vertex_label = 2
        for branch in range(self.m):
            previous_vertex = self.center
            for i in range(self.n):
                self.graph[vertex_label] = [previous_vertex]
                self.graph[previous_vertex].append(vertex_label)
                previous_vertex = vertex_label
                vertex_label += 1

    def assign_vertex_labels(self):
        label = 1
        for vertex in self.graph.keys():
            self.vertex_labels[vertex] = label
            label += 1

    def compute_edge_weights(self):
        visited_edges = set()
        for u in self.graph:
            for v in self.graph[u]:
                if (u, v) not in visited_edges and (v, u) not in visited_edges:
                    weight = self.vertex_labels[u] + self.vertex_labels[v]
                    self.edge_weights.append((u, v, weight))
                    visited_edges.add((u, v))

    def display_results(self):
        print("Vertex Labels:")
        for vertex, label in self.vertex_labels.items():
            print(f"Vertex {vertex}: Label {label}")

        print("\nEdge Weights:")
        for u, v, weight in self.edge_weights:
            print(f"Edge ({u}, {v}) -> Weight: {weight}")

    def display_table(self):
        print("\nTable of Vertices and Edge Weights:")
        print("| Vertex | Label | Neighbor | Neighbor's Label | Edge Weight (Sum) |")
        print("|--------|-------|----------|------------------|--------------------|")
        for u, v, weight in self.edge_weights:
            label_u = self.vertex_labels[u]
            label_v = self.vertex_labels[v]
            print(f"| {u} | {label_u} | {v} | {label_v} | {weight} |")

    def display_complexity(self):
        V = len(self.vertex_labels)
        E = len(self.edge_weights)
        print("\nEstimated Time Complexity:")
        print(f"Total Vertices V = {V}")
        print(f"Total Edges E = {E}")
        print(f"Theoretical Total Complexity T(V, E) ≈ O(V + E) = O({V} + {E}) = O({V + E})")

def k_labeling(n, m):
    vertex_labels = {}
    label = 1
    center_vertex = 1
    vertex_labels[center_vertex] = label
    label += 1
    current_vertex = 2
    for i in range(m):
        previous_vertex = center_vertex
        for j in range(n):
            vertex_labels[current_vertex] = label
            label += 1
            previous_vertex = current_vertex
            current_vertex += 1
    return vertex_labels

def bfs_traversal(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        print(f"Visited Node {node}")
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def dfs_traversal(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            print(f"Visited Node {node}")
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)

def test_max_resources():
    n = 10
    m = 10
    max_memory_usage = 16 * 1024 ** 3
    next_progress_threshold = 5
    while True:
        try:
            star_graph = StarGraph2(n, m)
            V = len(star_graph.vertex_labels)
            E = len(star_graph.edge_weights)
            memory_usage = psutil.Process().memory_info().rss
            memory_usage_mb = memory_usage / (1024 ** 2)
            progress = (memory_usage / max_memory_usage) * 100
            if progress >= next_progress_threshold:
                print(f"Progress: {next_progress_threshold}% - Current n: {n}, p: {m} - "
                      f"Vertices: {V}, Edges: {E}, Process memory usage: {memory_usage_mb:.2f} MB")
                next_progress_threshold += 5
            if memory_usage >= max_memory_usage:
                print("\nReached Maximum System Support:")
                print(f"Max Vertices V = {V}")
                print(f"Max Edges E = {E}")
                break
            m += 10
        except MemoryError:
            print("\nMemory limit reached, reached maximum system support scale.")
            print(f"Max Vertices V = {V}")
            print(f"Max Edges E = {E}")
            break

n = 3
m = 4
star_graph1 = StarGraph1(n, m)
star_graph1.assign_edge_weights()
star_graph1.display_graph()

vertex_labels = k_labeling(n, m)
print("Vertex Label Assignment:")
for vertex, label in vertex_labels.items():
    print(f"Vertex {vertex}: Label {label}")

print("\nBFS Traversal:")
bfs_traversal(star_graph1.graph, start=1)
print("\nDFS Traversal:")
dfs_traversal(star_graph1.graph, start=1)

star_graph2 = StarGraph2(n, m)
star_graph2.display_results()
star_graph2.display_table()
star_graph2.display_complexity()

test_max_resources()