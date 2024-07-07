import torch
import os
import sys
from tqdm import tqdm
references_path = os.path.join('..', 'references')
sys.path.append(references_path)
import dictionary

def feature_encoding(graph, exclude_centrality=False):
    """
    Encode the node attributes of a graph into a feature tensor, optionally excluding centrality metrics.

    Args:
    graph (networkx.Graph): The graph whose node attributes will be encoded.
    exclude_centrality (bool): If True, centrality metrics will be excluded from the encoding.

    Returns:
    tuple: A tuple containing:
        - torch.Tensor: The feature tensor with encoded node attributes.
        - list: The list of feature names corresponding to the tensor columns.
    """
    G = graph
    print(f"\nFeature encoding the node attributes...")

    # Function to safely convert the node attribute values from string to float
    def safe_float_convert(value):
        try:
            return float(value)
        except ValueError:
            return 0.0

    # Prepare node features and feature names
    node_features = []
    feature_names = []

    # Add tqdm progress bar
    for node_id in tqdm(G.nodes(), desc="Encoding node attributes"):
        # Encoding 'hostsld' or assigning 0 if it doesn't exist
        # Gathering and converting other attributes, excluding centrality metrics if specified
        if exclude_centrality:
            attributes = [(attr, safe_float_convert(G.nodes[node_id][attr])) for attr in G.nodes[node_id] if attr.lower() not in list(dictionary.centrality_metrics.keys())]
        else:
            attributes = [(attr, safe_float_convert(G.nodes[node_id][attr])) for attr in G.nodes[node_id]]
        
        # Append encoded 'hostsld' with other attributes
        node_features.append([attr[1] for attr in attributes])
        if not feature_names:
            feature_names = [attr[0] for attr in attributes]

    # Convert to tensor and handle NaNs
    feature_tensor = torch.tensor(node_features, dtype=torch.float)
    feature_tensor[torch.isnan(feature_tensor)] = 0

    print(f"Successfully feature encoded the node attributes")

    return feature_tensor, feature_names


