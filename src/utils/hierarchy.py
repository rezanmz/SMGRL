from anytree import NodeMixin
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj
from torch_geometric.data import Data
from pygsp.graphs.graph import Graph as gsp_Graph
from graph_coarsening import coarsen as graph_reduction
import torch


class Node(NodeMixin):
    def __init__(self, node_id, embedding=None, parent=None, children=None) -> None:
        super(Node, self).__init__()
        self.node_id = node_id
        self.embedding = embedding
        self.parent = parent
        if children is not None:
            self.children = children


def construct_hierarchy(
    data,
    coarsening_method,
    reduction_ratio,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj = to_scipy_sparse_matrix(data.edge_index)
    _, _, restriction_operators, _ = graph_reduction(
        gsp_Graph(adj),
        r=reduction_ratio,
        method=coarsening_method
    )
    # Convert restriction operators to torch
    restriction_operators = [torch.from_numpy(
        op.todense()).float().to(device) for op in restriction_operators]
    hierarchy = [[
        Node(node_id=node_id)
        for node_id in range(adj.shape[0])
    ]]
    graphs = []
    for restriction_operator in restriction_operators:
        # prolongation_operator = restriction_operator.transpose()
        partitions = [[] for _ in range(restriction_operator.shape[0])]
        train_y = []
        val_y = []
        for node in range(restriction_operator.shape[1]):
            partitions[restriction_operator[:, node].argmax()].append(node)
            train_y.append(
                data.train_y[node].tolist() if data.train_mask[node] else torch.zeros_like(
                    data.train_y[node]).tolist()
            )
            val_y.append(
                data.val_y[node].tolist() if data.val_mask[node] else torch.zeros_like(
                    data.val_y[node]).tolist()
            )

        subgraphs = []
        for partition in partitions:
            subgraphs.append({
                'nodes': partition,
                'data': data.subgraph(torch.tensor(partition).long()).to(device),
            })
        subgraphs.append({
            'nodes': [node for node in range(restriction_operator.shape[1])],
            'data': data.to(device),
        })
        graphs.append(subgraphs)

        train_y = torch.tensor(train_y).to(device)
        val_y = torch.tensor(val_y).to(device)

        train_y = restriction_operator.mm(train_y)
        val_y = restriction_operator.mm(val_y)
        features = restriction_operator.mm(data.x)
        adj = to_dense_adj(data.edge_index)[0]
        adj = restriction_operator.mm(adj).mm(restriction_operator.t())
        edge_index = adj.to_sparse().indices()
        data = Data(
            x=features,
            edge_index=edge_index,
            train_y=train_y,
            val_y=val_y,
            train_mask=torch.tensor([bool(train_y[i].sum())
                                    for i in range(adj.shape[0])]),
            val_mask=torch.tensor([bool(val_y[i].sum())
                                  for i in range(adj.shape[0])]),
        ).to(device)
        hierarchy.append([
            Node(
                node_id=node_id,
                children=[hierarchy[-1][child] for child in partition],
            )
            for node_id, partition in zip(range(data.x.shape[0]), partitions)
        ])

    graphs.append([{
        'nodes': [node for node in range(data.x.shape[0])],
        'data': data.to(device)
    }])
    return hierarchy, graphs


def aggregate(node):
    embedding = []
    while node.parent is not None:
        embedding.append(node.embedding.detach().cpu().numpy())
        node = node.parent
    results = [*embedding, node.embedding.detach().cpu().numpy()]
    return results
