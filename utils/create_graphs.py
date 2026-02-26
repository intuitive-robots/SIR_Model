import torch
import networkx as nx
from torch_geometric.data import Data

def create_graph_datapoint(graph: nx.Graph, 
                           object_names: list,
                           objects,
                           ) -> Data:
    if len(objects.shape) == 1:
        objects = objects.unsqueeze(0)
    add_nodes_to_graph(graph, object_names, objects)

    # Convert Graph to Data object
    adj = nx.to_scipy_sparse_array(graph).tocoo()
    row = adj.row
    col = adj.col
    nodes = []
    node_list = list(graph.nodes(data=True))
    node_list.sort(key=lambda tup: tup[0])

    for node_all in node_list:
        nodes.append(node_all[1]['object'])

    weights = []
    
    for edge in graph.edges(data=True):
        weights.append(edge[-1]['weight'])

    row_tensor = torch.tensor(row)
    col_tensor = torch.tensor(col)
    feature_tensor = torch.stack(nodes)
    if not isinstance(weights, torch.Tensor):
        weight_tensor = torch.tensor(weights)
    else:
        weight_tensor = weights

    graph_edge_index = torch.stack((col_tensor, row_tensor)) 

    data_point = Data(x=feature_tensor, edge_index=graph_edge_index, edge_attr=weight_tensor, node_names=[n for n, d in node_list])

    return data_point

# Adding nodes and updating them
def add_nodes_to_graph(Graph: nx.Graph, object_names: list, objects: list):
    for i, name in enumerate(object_names):
        if Graph.has_node(name):
            Graph.nodes[name]['object'] = objects[object_names.index(name)]
        else:
            Graph.add_node(name, object=objects[object_names.index(name)])

    for obj in list(Graph.nodes):
        for sub in list(Graph.nodes):
            if obj == sub:
                continue
            Graph.add_edge(obj, sub, weight=1)