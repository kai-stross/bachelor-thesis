import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
import os
import importlib
import sys
references_path = os.path.join('..', 'references')
sys.path.append(references_path)
import dictionary
import json

def load_graph(data_name, data_path):
    """
    Load a graph from GEXF and CSV files, adding new attributes from the CSV to the graph nodes.

    Args:
    data_name (str): The base name of the GEXF and CSV files (without extension).
    data_path (str): The directory path where the GEXF and CSV files are located.

    Returns:
    networkx.Graph: The graph with new attributes added to its nodes.
    """
    # The dataset has to be saved under the same name in both files in the data folder
    # Specify the GEXF and CSV file paths
    gexf_file_path = os.path.join(data_path, f"{data_name}.gexf")

    # Load the GEXF graph
    print(f"\nLoading graph from {gexf_file_path}...")
    G = nx.read_gexf(gexf_file_path)
    print(f"Successfully loaded graph from {gexf_file_path}")

    csv_file_path = os.path.join(data_path, f"{data_name}.csv")
    # Load new attributes from the CSV using centrality metrics from the dictionary
    print(f"Loading centrality metrics from {csv_file_path}...")
    new_attributes = pd.read_csv(
        csv_file_path,
        usecols=lambda x: x.lower() in list(dictionary.centrality_metrics.keys()) + ['id'], 
        index_col=False
    )
    new_attributes.columns = new_attributes.columns.str.lower()
    print(f"Successfully loaded centrality metrics from {csv_file_path}")

    # Add new attributes to the graph
    print(f"Adding new attributes to the graph...")
    for node_id, attributes in new_attributes.set_index('id').iterrows():
        if node_id in G:
            for attr_name, attr_value in attributes.items():
                G.nodes[node_id][attr_name] = attr_value
    print(f"Successfully added new attributes to the graph")

    # Return the updated graph
    return G

def remove_attributes_from_gexf(graph, config_path):
    """
    Remove unwanted attributes from a graph based on a configuration file.

    Args:
    graph (networkx.Graph): The graph from which attributes will be removed.
    config_path (str): The path to the configuration file that lists excluded features.

    Returns:
    networkx.Graph: The graph with unwanted attributes removed.
    """
    G = graph
    # Extract the directory and module name from the config_path
    references_path = os.path.dirname(config_path)
    config_module_name = os.path.basename(config_path).replace('.py', '')

    # Modify the system path to include the references directory if not already included
    if references_path not in sys.path:
        sys.path.append(references_path)

    # Dynamically import the configuration module
    print(f"\nImporting configuration from {config_path}...")
    config = importlib.import_module(config_module_name)
    print(f"Successfully imported configuration from {config_path}")

    # Remove unwanted edge attributes
    for u, v in G.edges():
        G.edges[u, v].clear()

    # Remove unwanted node attributes
    print(f"Removing unwanted attributes from the graph with the configuration in {config_path}...")
    excluded_features = [feature.lower() for feature in config.excluded_features]

    for node in G.nodes:
        attrs_to_delete = [attr for attr in G.nodes[node] if attr.lower() in excluded_features]
        for attr in attrs_to_delete:
            del G.nodes[node][attr]

    print(f"Successfully removed unwanted attributes from the graph with the configuration in {config_path}")

    # Return the updated graph
    return G

def generate_labels(graph):
    """
    Generate binary labels for the nodes in the graph based on the 'tracking' attribute.

    Args:
    graph (networkx.Graph): The graph whose nodes will be labeled.

    Returns:
    list: A list of binary labels for the nodes.
    """
    G = graph
    labels = []
    print(f"\nGenerating binary labels for the nodes in the graph...")

    # Iterate over all nodes in the graph
    for node_id in G.nodes:
        # Get the 'tracking' attribute, use 0 as a default if 'tracking' is not found
        tracking_value = float(G.nodes[node_id].get('tracking', 0))  
        # Append 1 if 'tracking' is at least 0.5, else append 0
        labels.append(1 if tracking_value >= 0.5 else 0)
    
    print(f"Successfully generated the binary labels for the nodes in the graph")

    return labels

def check_non_convertible_attributes(G):
    """
    Identify attributes in the graph nodes that cannot be converted to float.

    Args:
    G (networkx.Graph): The graph whose node attributes will be checked.

    Returns:
    list: A list of attribute names that cannot be converted to float.
    """
    non_convertible = set()  # Attributes that cannot be converted to float

    # Iterate over all nodes and their attributes
    for _, attrs in G.nodes(data=True):
        for attr, value in attrs.items():
            try:
                # Attempt to convert each attribute to float
                float(value)
            except ValueError:
                non_convertible.add(attr)  # Add to non-convertible if error occurs

    return list(non_convertible)

def networkx_to_pyg(graph, feature_tensor, labels):
    """
    Convert a NetworkX graph to a PyG (PyTorch Geometric) Data object.

    Args:
    graph (networkx.Graph): The NetworkX graph to be converted.
    feature_tensor (torch.Tensor): A tensor containing the node features.
    labels (list): A list of labels for the nodes.

    Returns:
    tuple: A tuple containing the PyG Data object and a dictionary mapping node indices to node names.
    """
    G = graph
    print(f"\nConverting NetworkX graph to PyG Data object...")
    # Convert labels to a tensor
    label_tensor = torch.tensor(labels, dtype=torch.long)

    # Map each unique node to an integer index and prepare edge list
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}  # Reverse mapping
    node_names = idx_to_node

    edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create the PyG Data object with all attributes
    data = Data(x=feature_tensor, edge_index=edge_index, num_nodes=len(G.nodes()))
    data.y = label_tensor

    print(f"Successfully converted NetworkX graph to PyG Data object")

    return data, node_names

def assign_masks(data, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    """
    Assign train, validation, and test masks to the PyG Data object.

    Args:
    data (torch_geometric.data.Data): The PyG Data object to which masks will be assigned.
    train_frac (float): The fraction of nodes to be used for training.
    val_frac (float): The fraction of nodes to be used for validation.
    test_frac (float): The fraction of nodes to be used for testing.

    Returns:
    torch_geometric.data.Data: The PyG Data object with assigned masks.
    """
    # Check if the fractions sum to 1
    if train_frac + val_frac + test_frac != 1:
        raise ValueError("The sum of train_frac, val_frac, and test_frac must equal 1.")

    print(f"\nAssigning masks to the PyG Data object...")
    num_nodes = data.num_nodes
    indices = np.random.permutation(num_nodes)
    train_end = int(train_frac * num_nodes)
    val_end = train_end + int(val_frac * num_nodes)

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)

    print(f"Successfully assigned masks to the PyG Data object")

    return data

def save_pyg(data, feature_names, node_names, save_path, save_metadata):
    """
    Save the PyG Data object and its metadata to files.

    Args:
    data (torch_geometric.data.Data): The PyG Data object to be saved.
    feature_names (list): The list of feature names.
    node_names (dict): A dictionary mapping node indices to node names.
    save_path (str): The file path to save the PyG Data object.
    save_metadata (str): The file path to save the metadata.

    Returns:
    None
    """
    print(f"\nSaving PyG Data object to {save_path}...")
    torch.save(data, save_path)
    print(f"Successfully saved PyG Data object to {save_path}: {data}")

    print(f"Saving PyG Data metadata to {save_metadata}...")
    metadata = {
        'node_names': node_names,
        'feature_names': feature_names
    }
    with open(save_metadata, 'w') as f:
        json.dump(metadata, f)
    print(f"Successfully saved metadata to {save_metadata}")
