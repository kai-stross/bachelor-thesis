import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from netgraph import Graph
import random
from math import sqrt
import os
import sys
sys.path.append('../references/')
import dictionary

# Function to show basic graph information of a networkx graph
def graph_info(graph):
    """
    Display basic information about a NetworkX graph.

    Args:
    graph (networkx.Graph): The graph for which information will be displayed.

    Returns:
    None
    """
    first_node = next(iter(graph.nodes))
    node_attributes = graph.nodes[first_node]
    node_attributes_list = list(node_attributes.keys())
    u, v, edge_attributes = next(iter(graph.edges(data=True)))
    edge_attributes_list = list(edge_attributes.keys())
    print(f"Directed Graph? {nx.is_directed(graph)} | Nodes: {nx.number_of_nodes(graph)} | Edges: {nx.number_of_edges(graph)} | Node attributes: {len(node_attributes_list)} | Edge attributes: {len(edge_attributes_list)}")
    print(f"Node attributes: {', '.join(node_attributes_list)}")
    print(f"Edge attributes: {', '.join(edge_attributes_list)}")

# Function to show node information of a networkx graph
def node_info(graph, n):
    """
    Display information for the first n nodes in a NetworkX graph.

    Args:
    graph (networkx.Graph): The graph containing the nodes.
    n (int): The number of nodes to display information for.

    Returns:
    None
    """
    for i, (node, attr) in enumerate(graph.nodes(data=True)):
        if i < n:  # Only print the first n nodes
            print(f"Node {node}: {attr}")
        else:
            break

# Function to get a subgraph from the first nodes, random nodes, with the top nodes of a centrality metric from the dictionary or proportionally from different communities from the dictionary
def create_subgraph(graph, criterion='first', num_nodes=1000):
    """
    Create a subgraph from a NetworkX graph based on specified criterion.

    Args:
    graph (networkx.Graph): The original graph.
    criterion (str): The criterion for selecting nodes ('first', 'random', centrality metric, or community metric).
    num_nodes (int): The number of nodes to include in the subgraph.

    Returns:
    tuple: A tuple containing the subgraph and its name.
    """
    G = graph
    total_nodes = len(G)
    if num_nodes > total_nodes:
        num_nodes = total_nodes

    nodes = list(G.nodes())
    if criterion == 'first':
        selected_nodes = nodes[:num_nodes]
        subgraph_name = 'Subgraph'
    
    elif criterion == 'random':
        selected_nodes = random.sample(nodes, num_nodes)
        subgraph_name = 'Random Subgraph'
       
    elif criterion in list(dictionary.community_metrics.keys()):
        subgraph_name = f'Sampled "{dictionary.community_metrics[criterion]}" Subgraph'
        community_assignments = {node: G.nodes[node][criterion] for node in nodes}
        communities = {}
        for node, comm_id in community_assignments.items():
            if comm_id in communities:
                communities[comm_id].append(node)
            else:
                communities[comm_id] = [node]

        selected_nodes = []
        total_community_nodes = sum(len(comm) for comm in communities.values())
        leftover_nodes = num_nodes
        for comm_nodes in communities.values():
            if leftover_nodes <= 0:
                break
            k = max(1, int(len(comm_nodes) / total_community_nodes * num_nodes))
            k = min(k, leftover_nodes)
            sampled = random.sample(comm_nodes, k)
            selected_nodes.extend(sampled)
            leftover_nodes -= len(sampled)

        all_sampled = set(selected_nodes)
        if leftover_nodes > 0:
            remaining_nodes = [node for node in nodes if node not in all_sampled]
            selected_nodes.extend(random.sample(remaining_nodes, leftover_nodes))
    
    elif criterion in list(dictionary.centrality_metrics.keys()):
        subgraph_name = f'Top "{dictionary.centrality_metrics[criterion]}" Subgraph'
        nodes_sorted_by_centrality = sorted(nodes, key=lambda x: G.nodes[x][criterion], reverse=True)
        selected_nodes = nodes_sorted_by_centrality[:num_nodes]

    else:
        raise ValueError("Criterion is not set to 'first' or 'random' and does not match any known node attribute. Check the dictionary for help!")
    
    subgraph = G.subgraph(selected_nodes)
    return subgraph, subgraph_name

# Function to draw a subgraph with a simple community layout, node color according to the label and node size according to a centrality metric
def draw_graph(graph, data_name, subgraph_criterion='first', nodes_amount=1000, labels=None, size_attr=None, community_attr=None, seed=None):
    """
    Draw a subgraph with node colors based on labels and sizes based on a centrality metric.

    Args:
    graph (networkx.Graph): The original graph.
    data_name (str): The name of the dataset.
    subgraph_criterion (str): The criterion for creating the subgraph ('first', 'random', etc.).
    nodes_amount (int): The number of nodes to include in the subgraph.
    labels (list): The list of labels for coloring nodes.
    size_attr (str): The centrality metric for determining node sizes.
    community_attr (str): The attribute for determining communities.
    seed (int): The seed for layout randomization.

    Returns:
    plt.Figure: The plotted figure.
    """
    subgraph, subgraph_name = create_subgraph(graph, criterion=subgraph_criterion, num_nodes=nodes_amount)
    if subgraph.number_of_nodes() < nodes_amount:
        subgraph_name = 'Entire Graph'

    color_map = ['skyblue' for _ in range(len(subgraph.nodes))]

    if labels is not None:
        for idx, node in enumerate(subgraph.nodes):
            if idx < len(labels):
                color_map[idx] = 'tab:red' if labels[idx] == 1 else 'tab:blue'
            else:
                break

    if community_attr:
        if community_attr in list(dictionary.community_metrics.keys()):
            communities = nx.get_node_attributes(graph, community_attr)
            unique_communities = list(set(communities.values()))
            grid_size = int(np.ceil(np.sqrt(len(unique_communities)))) * 5
            base_positions = {community: np.array([i % grid_size, i // grid_size]) * 3 for i, community in enumerate(unique_communities)}
            initial_pos = {node: base_positions[communities[node]] + 0.1 * np.random.randn(2) for node in subgraph.nodes()}
            pos = nx.spring_layout(subgraph, pos=initial_pos, k=5, scale=2, iterations=50)
        else:
            print(f"Error: '{community_attr}' is not a valid community metric for visualization. Check the dictionary for help!")
    else:
        pos = nx.spring_layout(subgraph, seed=seed)

    if size_attr is not None:
        if size_attr in list(dictionary.centrality_metrics.keys()):
            node_sizes = [graph.nodes[node].get(size_attr, 1) for node in subgraph.nodes]
            scaled_sizes = np.interp(node_sizes, (min(node_sizes), max(node_sizes)), (10, 1500))
            graph_title = f'{subgraph_name} - \n {subgraph.number_of_nodes()} Nodes - "{dictionary.centrality_metrics[size_attr]}" Node Size'
        else:
            print(f"Error: '{size_attr}' is not a valid centrality metric for visualization. Check the dictionary for help!")
            return
    else:
        scaled_sizes = 50
        graph_title = f'{subgraph_name} - {subgraph.number_of_nodes()} Nodes'

    fig, ax = plt.subplots(figsize=(10, 10))

    nx.draw(subgraph, pos, node_color=color_map, with_labels=False, node_size=scaled_sizes, font_size=8)

    if labels is not None:
        red_patch = mpatches.Patch(color='tab:red', label='Tracker')
        blue_patch = mpatches.Patch(color='tab:blue', label='Non-Tracker')
        plt.legend(handles=[red_patch, blue_patch])

    plt.title(f"{data_name}\n{graph_title}", fontsize=16)
    plt.show()
    
    return fig

# Function to draw a subgraph with a simple community layout, node color according to the community, node shape according to the label and node size according to a centrality metric
def draw_graph_communities(graph, data_name, community_attr, subgraph_criterion=None, nodes_amount=1000, labels=None, size_attr=None, seed=None):
    """
    Draw a subgraph with nodes colored by community, shaped by label, and sized by a centrality metric.

    Args:
    graph (networkx.Graph): The original graph.
    data_name (str): The name of the dataset.
    community_attr (str): The attribute for determining communities.
    subgraph_criterion (str): The criterion for creating the subgraph ('first', 'random', etc.).
    nodes_amount (int): The number of nodes to include in the subgraph.
    labels (list): The list of labels for determining node shapes.
    size_attr (str): The centrality metric for determining node sizes.
    seed (int): The seed for layout randomization.

    Returns:
    plt.Figure: The plotted figure.
    """
    subgraph, subgraph_name = create_subgraph(graph, criterion=subgraph_criterion, num_nodes=nodes_amount)
    graph_title = f'{subgraph_name} - {subgraph.number_of_nodes()} Nodes'

    fig, ax = plt.subplots()

    communities = nx.get_node_attributes(graph, community_attr)
    unique_communities = list(set(communities.values()))
    grid_size = int(np.ceil(np.sqrt(len(unique_communities)))) * 5
    base_positions = {community: np.array([i % grid_size, i // grid_size]) * 3 for i, community in enumerate(unique_communities)}
    initial_pos = {node: base_positions[communities[node]] + 0.1 * np.random.randn(2) for node in subgraph.nodes()}
    pos = nx.spring_layout(subgraph, pos=initial_pos, k=5, scale=2, iterations=50)

    color_map = []
    node_shapes = {}

    if community_attr in list(dictionary.community_metrics.keys()):
        communities = nx.get_node_attributes(subgraph, community_attr)
        unique_communities = list(set(communities.values()))
        color_palette = plt.cm.get_cmap('hsv', len(unique_communities))
        community_color = {community: color_palette(i) for i, community in enumerate(unique_communities)}
        color_map = [community_color[communities[node]] for node in subgraph.nodes()]
        graph_title = f'{subgraph_name} - \n {subgraph.number_of_nodes()} Nodes - "{dictionary.community_metrics[community_attr]}" Communities '
    else:
        print(f"Error: '{community_attr}' is not a valid community metric for visualization. Check the dictionary for help!")
        color_map = ['skyblue'] * len(subgraph.nodes())

    if labels is not None:
        for idx, node in enumerate(subgraph.nodes()):
            if idx < len(labels) and labels[idx] == 1:
                node_shapes[node] = 'v'
            else:
                node_shapes[node] = 'o'

    scaled_sizes = [50] * subgraph.number_of_nodes() 
    if size_attr is not None:
        if size_attr in list(dictionary.centrality_metrics.keys()):
            node_sizes = [graph.nodes[node].get(size_attr, 1) for node in subgraph.nodes()]
            scaled_sizes = np.interp(node_sizes, (min(node_sizes), max(node_sizes)), (10, 1500))
            graph_title += f'- "{dictionary.centrality_metrics[size_attr]}" Node Size'
        else:
            print(f"Error: '{size_attr}' is not a valid centrality metric for visualization. Check the dictionary for help!")
            return

    for node_shape in set(node_shapes.values()):
        shaped_nodes = [node for node in subgraph.nodes() if node_shapes[node] == node_shape]
        shaped_sizes = [scaled_sizes[i] for i, node in enumerate(subgraph.nodes()) if node_shapes[node] == node_shape]
        nx.draw_networkx_nodes(subgraph, pos, nodelist=shaped_nodes,
                            node_color=[color_map[i] for i, node in enumerate(subgraph.nodes()) if node_shapes[node] == node_shape],
                            node_shape=node_shape, node_size=shaped_sizes, label='Tracker' if node_shape == 'v' else 'Non-Tracker')

    nx.draw_networkx_edges(subgraph, pos)

    plt.legend(markerscale=0.35)
    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    leg = ax.get_legend()
    leg.legend_handles[0].set_color('black')
    if len(leg.legend_handles) > 1:
        leg.legend_handles[1].set_color('black')
    plt.gcf().set_size_inches(10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"{data_name}\n{graph_title}", fontsize=16)
    plt.show()

    return fig

# Function to draw a subgraph with a more complex but more clear community layout, node color according to the community, node shape according to the label and node size according to a centrality metric
# The function has a higher runtime so we choose less nodes to visualize
def draw_graph_netgraph_communities(graph, data_name, community_attr=None, subgraph_criterion=None, nodes_amount=500, labels=None, size_attr=None, seed=None):
    """
    Draw a subgraph with a detailed community layout using netgraph, with nodes colored by community, shaped by label, and sized by a centrality metric.

    Args:
    graph (networkx.Graph): The original graph.
    data_name (str): The name of the dataset.
    community_attr (str): The attribute for determining communities.
    subgraph_criterion (str): The criterion for creating the subgraph ('first', 'random', etc.).
    nodes_amount (int): The number of nodes to include in the subgraph.
    labels (list): The list of labels for determining node shapes.
    size_attr (str): The centrality metric for determining node sizes.
    seed (int): The seed for layout randomization.

    Returns:
    plt.Figure: The plotted figure.
    """
    subgraph, subgraph_name = create_subgraph(graph, criterion=subgraph_criterion, num_nodes=nodes_amount)
    graph_title = f'{subgraph_name} - \n {subgraph.number_of_nodes()} Nodes'

    if community_attr is not None:
        communities = nx.get_node_attributes(subgraph, community_attr)
        unique_communities = list(set(communities.values()))
        if community_attr in list(dictionary.community_metrics.keys()):
            color_palette = plt.cm.get_cmap('hsv', len(unique_communities))
            community_color = {community: color_palette(i) for i, community in enumerate(unique_communities)}
            node_color = {node: community_color[communities[node]] for node in subgraph.nodes()}
            graph_title += f' - "{dictionary.community_metrics[community_attr]}" Communities'
        else:
            print(f"Error: '{community_attr}' is not a valid community metric for visualization. Check the dictionary for help!")
            return
    else:
        undirected_subgraph = subgraph.to_undirected()
        communities = list(nx.community.label_propagation_communities(undirected_subgraph))
        communities_dict = {node: idx for idx, community in enumerate(communities) for node in community}
        color_palette = plt.cm.get_cmap('hsv', len(communities))
        node_color = {node: color_palette(communities_dict[node]) for node in subgraph.nodes()}
        graph_title += ' - "Label Propagation" Communities'

    node_sizes = {node: 1 for node in subgraph.nodes()}
    node_shapes = {node: 'o' for node in subgraph.nodes()}

    if size_attr is not None:
        if size_attr in list(dictionary.centrality_metrics.keys()):
            sizes = [graph.nodes[node].get(size_attr, 1) for node in subgraph.nodes()]
            node_sizes = np.interp(sizes, (min(sizes), max(sizes)), (1, 3))
            node_sizes = {node: size for node, size in zip(subgraph.nodes(), node_sizes)}
            graph_title += f' - "{dictionary.centrality_metrics[size_attr]}" Node Size'
    elif size_attr is not None:
        print(f"Error: '{size_attr}' is not a valid centrality metric for visualization. Check the dictionary for help!")
        return

    if labels is not None:
        for idx, node in enumerate(graph.nodes()):
            if idx < len(labels) and labels[idx] == 1:
                node_shapes[node] = 'v'

    fig, ax = plt.subplots()

    Graph(subgraph,
          node_color=node_color,
          node_size=node_sizes,
          node_shape=node_shapes,
          node_edge_width=0,
          edge_alpha=0.1,
          node_layout='community',
          node_layout_kwargs={'node_to_community': communities_dict if community_attr is None else communities},
          edge_layout='bundled',
          edge_layout_kwargs={'k': 2000}
    )

    plt.title(f"{data_name}\n{graph_title}", fontsize=16)
    plt.gcf().set_size_inches(10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10, label='Non-Tracker'),
                        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='k', markersize=10, label='Tracker')],
                        title_fontsize='13', fontsize='12', loc='upper right')
    plt.show()

    return fig

def draw_domain_neighbors(graph, data_name, domain_names, labels, depth=1):
    """
    Plot subgraphs for specified domain names from a NetworkX directed graph.

    Args:
    graph (networkx.DiGraph): The NetworkX directed graph.
    domain_names (list): The list of domain names for which to plot subgraphs.
    labels (list): The list of labels for each node (0 or 1).
    depth (int): The depth of neighbors to include in the subgraph.

    Returns:
    list: A list of tuples containing the plotted figures and domain names.
    """
    # Create a mapping from node names to indices
    node_to_index = {node: idx for idx, node in enumerate(graph.nodes)}
    
    # Define a helper function to visualize the subgraph
    def visualize_custom_graph(subgraph, domain_name):
        node_size = 800

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(subgraph)
        
        for src, dst, data in subgraph.edges(data=True):
            ax.annotate(
                '',
                xy=pos[src],
                xytext=pos[dst],
                arrowprops=dict(
                    arrowstyle="->",
                    alpha=data.get('weight', 1.0),
                    shrinkA=sqrt(node_size) / 2.0,
                    shrinkB=sqrt(node_size) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

        node_colors = ['white' for _ in subgraph.nodes()]
        node_edgecolors = ['red' if labels[node_to_index[node]] == 1 else 'blue' for node in subgraph.nodes()]
        node_edgewidths = [2 if node == domain_name else 1 for node in subgraph.nodes()]

        nx.draw(subgraph, pos, with_labels=False, node_size=node_size, node_color=node_colors, edgecolors=node_edgecolors, linewidths=node_edgewidths, ax=ax)

        for node, (x, y) in pos.items():
            fontweight = 'bold' if node == domain_name else 'normal'
            label = node
            plt.text(x, y, s=label, fontsize=10, fontdict={'color': 'black', 'weight': fontweight}, horizontalalignment='center', verticalalignment='center')

        # Add legend
        red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Tracker', markersize=10, markerfacecolor='white', markeredgecolor='red')
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Non-Tracker', markersize=10, markerfacecolor='white', markeredgecolor='blue')
        plt.legend(handles=[red_patch, blue_patch])

        plt.title(f"{data_name}\nSubgraph for Node \"{domain_name}\"")
        plt.show()

        return fig
    
    def get_subgraph_nodes(graph, start_node, depth):
        visited = set()
        queue = [(start_node, 0)]
        while queue:
            current_node, current_depth = queue.pop(0)
            if current_depth > depth:
                continue
            if current_node not in visited:
                visited.add(current_node)
                neighbors = list(graph.successors(current_node)) + list(graph.predecessors(current_node))
                for neighbor in neighbors:
                    queue.append((neighbor, current_depth + 1))
        return visited
    
    figs = []
    for domain_name in domain_names:
        if domain_name not in graph.nodes:
            print(f"Domain name {domain_name} not found in the node names.")
            continue

        # Get subgraph nodes up to the specified depth
        subgraph_nodes = get_subgraph_nodes(graph, domain_name, depth)
        subgraph = graph.subgraph(subgraph_nodes)

        # Visualize the custom graph
        fig = visualize_custom_graph(subgraph, domain_name)
        figs.append((fig, domain_name))
    
    return figs


def save_report_figure(fig, data_name, file_name):
    """
    Save a matplotlib figure to a file.

    Args:
    fig (plt.Figure): The figure to be saved.
    data_name (str): The name of the dataset.
    file_name (str): The name of the file.

    Returns:
    None
    """
    folder = os.path.join('..', 'reports', 'figures', data_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, f'{file_name}.png')
    fig.savefig(file_path)
