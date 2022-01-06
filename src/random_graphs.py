import argparse
import gc
import json
import os
from time import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import models
from utils.datasets import random_graph
from utils.hierarchy import construct_hierarchy


def baseline_runtime(
    graph_type,
    num_nodes,
    gnn_model,
    embedding_dim,
    num_epochs,
):
    print(
        f'Baseline - GNN Model: {gnn_model}, Graph Type: {graph_type}, Num Nodes: {num_nodes}'
    )
    # Load dataset
    data, num_classes, num_features = random_graph(
        type=graph_type, size=num_nodes)

    if gnn_model == 'GraphSAGE':
        model = models.GraphSAGE(num_features, num_classes, embedding_dim)
    elif gnn_model == 'APPNP':
        model = models.APPNP(num_features, num_classes, embedding_dim)

    optimizer = torch.optim.RMSprop(model.parameters())

    # Time the training
    start = time()

    # Train model
    model.train()
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        optimizer.zero_grad()
        _, logits = model(data)
        loss = F.cross_entropy(logits[data.train_mask],
                               data.train_y[data.train_mask])
        loss.backward()
        optimizer.step()

    end = time()

    baseline_time = end - start
    print(f'Baseline Runtime: {baseline_time}')
    return {
        'time': baseline_time,
    }


def dmgrl_runtime(
    graph_type,
    num_nodes,
    gnn_model,
    embedding_dim,
    num_epochs,
    reduction_ratio,
    coarsening_method,
):
    print(
        f'DMGRL - GNN Model: {gnn_model}, Graph Type: {graph_type}, Num Nodes: {num_nodes}'
    )
    # Load dataset
    data, num_classes, num_features = random_graph(
        type=graph_type, size=num_nodes)

    # Construct hierarchy
    start = time()
    hierarchy, hierarchy_graphs = construct_hierarchy(
        data,
        coarsening_method,
        reduction_ratio,
    )
    end = time()

    results = {
        'hierarchy_construction_time': end - start,
        'coarse_level_training': None,
        'hierarchy_eval': {
            f'level{i}': []
            for i in range(len(hierarchy))
        }
    }

    if gnn_model == 'GraphSAGE':
        model = models.GraphSAGE(num_features, num_classes, embedding_dim)
    elif gnn_model == 'APPNP':
        model = models.APPNP(num_features, num_classes, embedding_dim)

    optimizer = torch.optim.RMSprop(model.parameters())

    # Time the training
    start = time()

    # Train the model on the coarsest level
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters())
    for epoch in tqdm(range(num_epochs), total=num_epochs, desc='Training'):
        optimizer.zero_grad()
        _, logits = model(hierarchy_graphs[-1][0]['data'])
        loss = F.cross_entropy(logits[hierarchy_graphs[-1][0]['data'].train_mask],
                               hierarchy_graphs[-1][0]['data'].train_y[hierarchy_graphs[-1][0]['data'].train_mask])
        loss.backward()
        optimizer.step()
    end = time()
    results['coarse_level_training'] = end - start

    # Evaluate
    pbar = tqdm(enumerate(hierarchy_graphs),
                desc='Evaluating', total=len(hierarchy))
    model.eval()
    for level, level_graphs in pbar:
        if len(level_graphs) > 1:
            _graphs = level_graphs[:-1]
        else:
            _graphs = level_graphs
        for i, graph in enumerate(_graphs):
            start = time()
            model(graph['data'])
            end = time()
            results['hierarchy_eval'][f'level{level}'].append(end - start)
            # Release unused memory
            gc.collect()
            pbar.set_description(
                f'Evaluating - Level {level}, Graph {i}/{len(_graphs)}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='dmgrl')
    parser.add_argument('--graph_type', type=str, default='erdos-renyi')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--gnn_model', type=str, default='GraphSAGE')
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--reduction_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str,
                        default='variation_neighborhoods')

    args = parser.parse_args()

    if args.method == 'baseline':
        results = baseline_runtime(
            args.graph_type,
            args.num_nodes,
            args.gnn_model,
            args.embedding_dim,
            args.num_epochs,
        )
        file_name = f'runtime_results/baseline/graph_type/{args.graph_type}/gnn_model/{args.gnn_model}/num_nodes/{args.num_nodes}/embedding_dim/{args.embedding_dim}/num_epochs/{args.num_epochs}/'
    elif args.method == 'dmgrl':
        results = dmgrl_runtime(
            args.graph_type,
            args.num_nodes,
            args.gnn_model,
            args.embedding_dim,
            args.num_epochs,
            args.reduction_ratio,
            args.coarsening_method,
        )
        file_name = f'runtime_results/dmgrl/graph_type/{args.graph_type}/gnn_model/{args.gnn_model}/num_nodes/coarsening_method/{args.coarsening_method}/reduction_ratio/{args.reduction_ratio}/{args.num_nodes}/embedding_dim/{args.embedding_dim}/num_epochs/{args.num_epochs}/'
    os.makedirs(file_name, exist_ok=True)
    num_files = len(os.listdir(file_name))
    with open(f'{file_name}/{num_files}.json', 'w') as f:
        json.dump(results, f)
