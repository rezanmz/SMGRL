import argparse
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from models import models
from utils.classify import classify
from utils.datasets import load_dataset
from utils.hierarchy import Node, construct_hierarchy

AGGREGATION_METHODS = ['mean', 'post_hoc_learned_weights', 'concat']
EARLY_STOPPING_PATIENCE = 5


def inductive(
    dataset_name,
    held_out_fraction,
    coarsening_method,
    reduction_ratio,
    gnn_model,
    embedding_dim,
    full_graph_per_level,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load dataset
    print('Loading dataset...')
    data, num_classes, num_features = load_dataset(dataset_name)
    print('Dataset loaded.')
    print('Removing held out nodes...')
    # Select held out nodes randomly
    held_out_nodes = np.random.choice(
        data.num_nodes,
        size=int(held_out_fraction * data.num_nodes),
        replace=False
    )
    # Remaining nodes
    remaining_nodes = np.setdiff1d(
        np.arange(data.num_nodes),
        held_out_nodes
    )
    # Get the subgraph for remaining nodes
    remaining_data = data.subgraph(torch.tensor(remaining_nodes))
    print('Held out nodes removed.')

    # Label all the remaining nodes as traning nodes
    # remaining_data.train_mask = torch.ones(
    #     remaining_data.num_nodes, dtype=torch.bool)
    # remaining_data.val_mask = torch.zeros(
    #     remaining_data.num_nodes, dtype=torch.bool)
    # remaining_data.test_mask = torch.zeros(
    #     remaining_data.num_nodes, dtype=torch.bool)

    # Construct hierarchy
    print('Constructing hierarchy...')
    hierarchy, hierarchy_graphs = construct_hierarchy(
        remaining_data,
        coarsening_method,
        reduction_ratio,
    )
    print('Hierarchy constructed.')

    # Initialize the model
    if gnn_model == 'GraphSAGE':
        model = models.GraphSAGE(
            num_features, num_classes, embedding_dim).to(device)
    elif gnn_model == 'APPNP':
        model = models.APPNP(
            num_features, num_classes, embedding_dim).to(device)

    # Train the model on the coarsest graph
    print('Training model on the coarsest graph...')
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters())
    current_patience = EARLY_STOPPING_PATIENCE
    best_val_loss = None
    best_model = None
    model.train()
    while True:
        optimizer.zero_grad()
        _, logits = model(hierarchy_graphs[-1][0]['data'].to(device))
        loss = F.cross_entropy(
            logits[hierarchy_graphs[-1][0]['data'].to(device).train_mask], hierarchy_graphs[-1][0]['data'].to(device).train_y[hierarchy_graphs[-1][0]['data'].to(device).train_mask])
        loss.backward()
        optimizer.step()
        val_loss = F.cross_entropy(
            logits[hierarchy_graphs[-1][0]['data'].to(device).val_mask], hierarchy_graphs[-1][0]['data'].to(device).train_y[hierarchy_graphs[-1][0]['data'].to(device).val_mask])
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            current_patience = EARLY_STOPPING_PATIENCE
        else:
            current_patience -= 1
            if current_patience == 0:
                break
    print('Training completed.')
    # Evaluate
    print('Generating embeddings for the remaining nodes (embedding1)...')
    model.load_state_dict(best_model)
    model.eval()
    for level_nodes, level_graphs in zip(hierarchy, hierarchy_graphs):
        if full_graph_per_level:
            _graphs = [level_graphs[-1]]
        elif not full_graph_per_level and len(level_graphs) > 1:
            _graphs = level_graphs[:-1]
        else:
            _graphs = level_graphs
        for graph in _graphs:
            embedding, _ = model(graph['data'].to(device))
            # Release unused memory
            gc.collect()
            torch.cuda.empty_cache()

            for node_idx, node in enumerate(graph['nodes']):
                level_nodes[node].embedding = embedding[node_idx]
    print('Embeddings generated.')

    # Find held out neighbors in the remaining nodes data
    print('Adding held out nodes...')
    held_out_connections_to_remaining_nodes = {}
    held_out_train_mask = []
    held_out_val_mask = []
    held_out_test_mask = []
    held_out_labels = []
    for held_out_node in held_out_nodes:
        held_out_connections_to_remaining_nodes[held_out_node] = {
            'neighbors': set(), 'subgraph': None}
        for e1, e2 in data.edge_index[:, torch.bitwise_or(data.edge_index[0, :] == held_out_node, data.edge_index[1, :] == held_out_node)].t().detach().numpy():
            if e1 == held_out_node and e2 in remaining_nodes:
                held_out_connections_to_remaining_nodes[held_out_node]['neighbors'].add(
                    int(np.argwhere(remaining_nodes == e2)))
            elif e2 == held_out_node and e1 in remaining_nodes:
                held_out_connections_to_remaining_nodes[held_out_node]['neighbors'].add(
                    int(np.argwhere(remaining_nodes == e1)))
        # Determine where the held out nodes are in the first level subgraphs (where they have most edges)
        num_connections = 0
        for idx, subgraph in enumerate(hierarchy_graphs[0][:-1]):
            if len(held_out_connections_to_remaining_nodes[held_out_node]['neighbors'] & set(subgraph['nodes'])) > num_connections:
                num_connections = len(
                    held_out_connections_to_remaining_nodes[held_out_node]['neighbors'] & set(subgraph['nodes']))
                held_out_connections_to_remaining_nodes[held_out_node]['subgraph'] = idx
        # Add the held out node to hierarchy
        if held_out_connections_to_remaining_nodes[held_out_node]['subgraph'] is not None:
            if full_graph_per_level:
                subgraph = hierarchy_graphs[0][-1]
            else:
                subgraph = hierarchy_graphs[0][held_out_connections_to_remaining_nodes[held_out_node]['subgraph']]
            subgraph['nodes'] = [str(node) for node in subgraph['nodes']]
            for neighbor in held_out_connections_to_remaining_nodes[held_out_node]['neighbors']:
                if neighbor in subgraph['nodes']:
                    subgraph['data'].edge_index = torch.cat((subgraph['data'].edge_index, torch.tensor([
                        len(subgraph['nodes']), int(np.argwhere(
                            np.array(subgraph['nodes']) == str(neighbor)))
                    ]).reshape(2, -1)), 1)
            subgraph['data'].x = torch.cat(
                (subgraph['data'].x, data.x[held_out_node].reshape(1, -1).to(device)), 0)
            subgraph['data'].train_y = torch.cat(
                (subgraph['data'].train_y, data.train_y[held_out_node].reshape(1, -1).to(device)), 0)
            subgraph['data'].val_y = torch.cat(
                (subgraph['data'].val_y, data.val_y[held_out_node].reshape(1, -1).to(device)), 0)
            subgraph['nodes'].append(f'held_out_{len(hierarchy[0])}')
            hierarchy[0].append(
                Node(
                    f'held_out_{len(hierarchy[0])}',
                    parent=hierarchy[1][held_out_connections_to_remaining_nodes[held_out_node]['subgraph']],
                )
            )
            held_out_train_mask.append(bool(data.train_mask[held_out_node]))
            held_out_val_mask.append(bool(data.val_mask[held_out_node]))
            held_out_test_mask.append(bool(data.test_mask[held_out_node]))
            held_out_labels.append(int(data.train_y.argmax(dim=1)[
                                   held_out_node].detach().cpu()))
    print('Held out nodes added.')
    # Evaluate on the new hierarchy (including held out nodes)
    print('Generating embeddings for the new hierarchy (embedding2)...')
    for level_nodes, level_graphs in zip(hierarchy, hierarchy_graphs):
        if full_graph_per_level:
            _graphs = [level_graphs[-1]]
        elif not full_graph_per_level and len(level_graphs) > 1:
            _graphs = level_graphs[:-1]
        else:
            _graphs = level_graphs
        for graph in _graphs:
            embedding, _ = model(graph['data'].to(device))
            # Release unused memory
            gc.collect()
            torch.cuda.empty_cache()

            for node_idx, node in enumerate(graph['nodes']):
                if isinstance(node, str) and 'held_out_' in node:
                    level_nodes[int(node[len('held_out_'):])
                                ].embedding2 = embedding[node_idx]
                else:
                    level_nodes[int(node)].embedding2 = embedding[node_idx]
    print('Embeddings generated.')
    results = {}
    # Classify training nodes using embedding1
    print('Classifying training nodes using embedding1...')
    results['training_nodes_embedding1'] = {}
    training_nodes = [node for node in hierarchy[0]
                      if node.embedding is not None]
    results['num_training_nodes'] = len(training_nodes)
    for aggregation_method in AGGREGATION_METHODS:
        pred_labels = classify(
            training_nodes,
            aggregation_method=aggregation_method,
            level=None,
            num_classes=num_classes,
            num_levels=len(hierarchy),
            train_mask=remaining_data.train_mask.detach().cpu().numpy(),
            val_mask=remaining_data.val_mask.detach().cpu().numpy(),
            labels=remaining_data.train_y.argmax(dim=1).detach().cpu().numpy(),
            embedding_type='embedding1',
        )
        results['training_nodes_embedding1'][aggregation_method] = {
            'accuracy': accuracy_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
            ),
            'f1_macro': f1_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
                average='macro'
            ),
            'f1_micro': f1_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
                average='micro'
            ),
        }
    for level in range(len(hierarchy)):
        pred_labels = classify(
            training_nodes,
            aggregation_method='level',
            level=level,
            num_classes=num_classes,
            num_levels=len(hierarchy),
            train_mask=remaining_data.train_mask.detach().cpu().numpy(),
            val_mask=remaining_data.val_mask.detach().cpu().numpy(),
            labels=remaining_data.train_y.argmax(dim=1).detach().cpu().numpy(),
            embedding_type='embedding1',
        )
        results['training_nodes_embedding1'][f'level{level}'] = {
            'accuracy': accuracy_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
            ),
            'f1_macro': f1_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
                average='macro'
            ),
            'f1_micro': f1_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
                average='micro'
            ),
        }
    print('Classification completed.')
    # Classify training nodes using embedding2
    print('Classifying training nodes using embedding2...')
    results['training_nodes_embedding2'] = {}
    training_nodes = [node for node in hierarchy[0]
                      if node.embedding is not None]
    for aggregation_method in AGGREGATION_METHODS:
        pred_labels = classify(
            training_nodes,
            aggregation_method=aggregation_method,
            level=None,
            num_classes=num_classes,
            num_levels=len(hierarchy),
            train_mask=remaining_data.train_mask.detach().cpu().numpy(),
            val_mask=remaining_data.val_mask.detach().cpu().numpy(),
            labels=remaining_data.train_y.argmax(dim=1).detach().cpu().numpy(),
            embedding_type='embedding2',
        )
        results['training_nodes_embedding2'][aggregation_method] = {
            'accuracy': accuracy_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()]
            ),
            'f1_macro': f1_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
                average='macro'
            ),
            'f1_micro': f1_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
                average='micro'
            ),
        }
    for level in range(len(hierarchy)):
        pred_labels = classify(
            training_nodes,
            aggregation_method='level',
            level=level,
            num_classes=num_classes,
            num_levels=len(hierarchy),
            train_mask=remaining_data.train_mask.detach().cpu().numpy(),
            val_mask=remaining_data.val_mask.detach().cpu().numpy(),
            labels=remaining_data.train_y.argmax(dim=1).detach().cpu().numpy(),
            embedding_type='embedding2',
        )
        results['training_nodes_embedding2'][f'level{level}'] = {
            'accuracy': accuracy_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
            ),
            'f1_macro': f1_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
                average='macro'
            ),
            'f1_micro': f1_score(
                remaining_data.train_y.argmax(dim=1).detach().cpu().numpy()[
                    remaining_data.test_mask.detach().cpu().numpy()],
                pred_labels[remaining_data.test_mask.detach().cpu().numpy()],
                average='micro'
            ),
        }
    print('Classification completed.')

    # Classify held out nodes using embedding2
    print('Classifying held out nodes using embedding2...')
    results['held_out_nodes_embedding2'] = {}
    held_out_nodes = [node for node in hierarchy[0]
                      if node.embedding2 is not None and node.embedding is None]
    results['num_held_out_nodes'] = len(held_out_nodes)
    for aggregation_method in AGGREGATION_METHODS:
        pred_labels = classify(
            held_out_nodes,
            aggregation_method=aggregation_method,
            level=None,
            num_classes=num_classes,
            num_levels=len(hierarchy),
            train_mask=np.array(held_out_train_mask),
            val_mask=np.array(held_out_val_mask),
            labels=np.array(held_out_labels),
            embedding_type='embedding2',
        )
        results['held_out_nodes_embedding2'][aggregation_method] = {
            'accuracy': accuracy_score(
                np.array(held_out_labels)[np.array(held_out_test_mask)],
                pred_labels[np.array(held_out_test_mask)]
            ),
            'f1_macro': f1_score(
                np.array(held_out_labels)[np.array(held_out_test_mask)],
                pred_labels[np.array(held_out_test_mask)],
                average='macro'
            ),
            'f1_micro': f1_score(
                np.array(held_out_labels)[np.array(held_out_test_mask)],
                pred_labels[np.array(held_out_test_mask)],
                average='micro'
            ),
        }
    for level in range(len(hierarchy)):
        pred_labels = classify(
            held_out_nodes,
            aggregation_method='level',
            level=level,
            num_classes=num_classes,
            num_levels=len(hierarchy),
            train_mask=np.array(held_out_train_mask),
            val_mask=np.array(held_out_val_mask),
            labels=np.array(held_out_labels),
            embedding_type='embedding2',
        )
        results['held_out_nodes_embedding2'][f'level{level}'] = {
            'accuracy': accuracy_score(
                np.array(held_out_labels)[np.array(held_out_test_mask)],
                pred_labels[np.array(held_out_test_mask)]
            ),
            'f1_macro': f1_score(
                np.array(held_out_labels)[np.array(held_out_test_mask)],
                pred_labels[np.array(held_out_test_mask)],
                average='macro'
            ),
            'f1_micro': f1_score(
                np.array(held_out_labels)[np.array(held_out_test_mask)],
                pred_labels[np.array(held_out_test_mask)],
                average='micro'
            ),
        }
    print('Classification completed.')

    # Classify all nodes using embedding2
    print('Classifying all nodes using embedding2...')
    results['overall_embedding2'] = {}
    train_mask = torch.concat((remaining_data.train_mask, torch.tensor(
        held_out_train_mask)), 0).detach().cpu().numpy()
    val_mask = torch.concat((remaining_data.val_mask, torch.tensor(
        held_out_val_mask)), 0).detach().cpu().numpy()
    test_mask = torch.concat((remaining_data.test_mask, torch.tensor(
        held_out_test_mask)), 0).detach().cpu().numpy()
    labels = torch.cat((remaining_data.train_y.argmax(dim=1), torch.tensor(
        held_out_labels)), 0).detach().cpu().numpy()
    for aggregation_method in AGGREGATION_METHODS:
        pred_labels = classify(
            hierarchy[0],
            aggregation_method=aggregation_method,
            level=None,
            num_classes=num_classes,
            num_levels=len(hierarchy),
            train_mask=train_mask,
            val_mask=val_mask,
            labels=labels,
            embedding_type='embedding2',
        )
        results['overall_embedding2'][aggregation_method] = {
            'accuracy': accuracy_score(labels[test_mask], pred_labels[test_mask]),
            'f1_macro': f1_score(
                labels[test_mask], pred_labels[test_mask],
                average='macro'
            ),
            'f1_micro': f1_score(
                labels[test_mask], pred_labels[test_mask],
                average='micro'
            ),
        }
    for level in range(len(hierarchy)):
        pred_labels = classify(
            hierarchy[0],
            aggregation_method='level',
            level=level,
            num_classes=num_classes,
            num_levels=len(hierarchy),
            train_mask=train_mask,
            val_mask=val_mask,
            labels=labels,
            embedding_type='embedding2',
        )
        results['overall_embedding2'][f'level{level}'] = {
            'accuracy': accuracy_score(labels[test_mask], pred_labels[test_mask]),
            'f1_macro': f1_score(
                labels[test_mask], pred_labels[test_mask],
                average='macro'
            ),
            'f1_micro': f1_score(
                labels[test_mask], pred_labels[test_mask],
                average='micro'
            ),
        }
    print('Classification completed.')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--held_out_fraction', type=float, default=0.1)
    parser.add_argument('--coarsening_method', type=str,
                        default='variation_neighborhoods')
    parser.add_argument('--reduction_ratio', type=float, default=0.5)
    parser.add_argument('--gnn_model', type=str, default='GraphSAGE')
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--full_graph_per_level',
                        action='store_true', dest='full_graph_per_level')
    parser.add_argument('--subgraphs_per_level',
                        action='store_false', dest='full_graph_per_level')

    args = parser.parse_args()

    results = inductive(
        dataset_name=args.dataset,
        held_out_fraction=args.held_out_fraction,
        coarsening_method=args.coarsening_method,
        reduction_ratio=args.reduction_ratio,
        gnn_model=args.gnn_model,
        embedding_dim=args.embedding_dim,
        full_graph_per_level=args.full_graph_per_level,
    )
    file_name = f'inductive_results/dataset/{args.dataset}/' \
                f'held_out_fraction/{args.held_out_fraction}/' \
                f'coarsening_method/{args.coarsening_method}/' \
                f'reduction_ratio/{args.reduction_ratio}/' \
                f'gnn_model/{args.gnn_model}/' \
                f'embedding_dim/{args.embedding_dim}/' \
                f'full_graph_per_level/{args.full_graph_per_level}/'
    os.makedirs(file_name, exist_ok=True)
    num_files = len(os.listdir(file_name))
    with open(f'{file_name}/{num_files}.json', 'w') as f:
        json.dump(results, f)
