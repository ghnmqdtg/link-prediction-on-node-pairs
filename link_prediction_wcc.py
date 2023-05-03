# %% [markdown]
# # Link Prediction Using WCC

# %% [markdown]
# 1. Import required packages and define functions

# %%
import os
import config
import shutil
import numpy as np
import pandas as pd
import networkx as nx
from tabulate import tabulate

# %%
def show_graph_info(graph):
    """Display graph information"""
    # Compute the number of nodes and edges of the graph
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    # Compute in-degree and out-degree of each node
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    # Compute average in-degree and out-degree
    avg_in_degree = sum(in_degrees.values()) / len(in_degrees)
    avg_out_degree = sum(out_degrees.values()) / len(out_degrees)
   # Create a table with the graph information
    table = [
        ["Number of nodes", num_nodes],
        ["Number of edges", num_edges],
        ["Average in-degree", f"{avg_in_degree:.2f}"],
        ["Average out-degree", f"{avg_out_degree:.2f}"]
    ]
    # Print the table
    print(tabulate(table, headers=["Metrices", "Value"]))

# %%
def compute_shortest_path_length(a, b, graph):
    """Computes the shortest path length"""
    p = -1
    try:
        if graph.has_edge(a, b):
            # Temporarily remove the edge to compute the shortest path length
            graph.remove_edge(a, b)
            p = nx.shortest_path_length(graph, source=a, target=b)
            graph.add_edge(a, b)
        else:
            p = nx.shortest_path_length(graph, source=a, target=b)
        return p
    except:
        return -1

# %%
def belongs_to_same_wcc(a, b, graph, wcc):
    """Check if two nodes belong to the same weakly connected component (WCC)"""
    index = []
    # If edge between a and b exists, return 1
    if graph.has_edge(b, a):
        return 1
    # If edge between b and a does not exist
    if graph.has_edge(a, b):
        for i in wcc:
            if a in i:
                index = i
                break
        # If both nodes are in the same WCC, return 1; otherwise, return 0
        if (b in index):
            graph.remove_edge(a, b)
            if compute_shortest_path_length(a, b, graph) == -1:
                graph.add_edge(a, b)
                return 0
            else:
                graph.add_edge(a, b)
                return 1
        else:
            return 0
    else:
        for i in wcc:
            if a in i:
                index = i
                break
        # If both nodes are in the same WCC, return 1; otherwise, return 0
        if (b in index):
            return 1
        else:
            return 0

# %% [markdown]
# 2. Load the training data and create the graph

# %%
# Load the csv file
df = pd.read_csv(config.TRAIN_CSV)
# Create the graph
G = nx.from_pandas_edgelist(df[df['label'] == 1], "node1", "node2", create_using=nx.DiGraph())
# Printing the information of graph
show_graph_info(G)

# %% [markdown]
# 3. Prepare WCC of training set, then apply to testing set

# %%
wcc = list(nx.weakly_connected_components(G))
df_test = pd.read_csv(config.TEST_CSV)
df_test['same_comp'] = df_test.apply(
    lambda row: belongs_to_same_wcc(row['node1'], row['node2'], G, wcc), axis=1)

# %% [markdown]
# 4. Export the predictions to `data/sample_submit.csv`

# %%
print(f"Number of the prediction [0, 1]: {np.bincount(df_test['same_comp'])}")
# Load the CSV file into a Pandas DataFrame
df_target = pd.read_csv(config.TARGET_CSV)
# Replace the values in the 'ans' column with the values from the Python list
df_target['ans'] = df_test['same_comp']
# Save the updated DataFrame back to a CSV file, overwriting the original file
df_target.to_csv(config.TARGET_CSV, index=False)


