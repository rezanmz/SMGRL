import argparse
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from models import models
from utils.classify import classify
from utils.datasets import load_dataset
from utils.hierarchy import construct_hierarchy

AGGREGATION_METHODS = ['mean', 'post_hoc_learned_weights', 'concat']
EARLY_STOPPING_PATIENCE = 5


def dmgrl(
    dataset_name,
    gnn_model,
    runs,
    embedding_dim,
    coarsening_method,
    reduction_ratio,
    full_graph_per_level
):
    print(
        f'DMGRL - Dataset: {dataset_name}, GNN Model: {gnn_model}, Embedding Dim: {embedding_dim}, Reduction Ratio: {reduction_ratio}, Full Graph Per Level: {full_graph_per_level}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load dataset
    data, num_classes, num_features = load_dataset(dataset_name)
    train_mask = data.train_mask.detach().cpu().numpy()
    val_mask = data.val_mask.detach().cpu().numpy()
    test_mask = data.test_mask.detach().cpu().numpy()
    labels = data.train_y.argmax(dim=1).detach().cpu().numpy()
    # Construct hierarchy
    hierarchy, hierarchy_graphs = construct_hierarchy(
        data,
        coarsening_method,
        reduction_ratio,
    )
    # Release unused memory
    gc.collect()
    torch.cuda.empty_cache()

    scores = {aggregation_method: {
        'accuracy': [],
        'f1_macro': [],
        'f1_micro': [],
    } for aggregation_method in AGGREGATION_METHODS}

    scores = {**scores, **{f'level{level}': {
        'accuracy': [],
        'f1_macro': [],
        'f1_micro': [],
    } for level in range(len(hierarchy))}}

    pbar = tqdm(range(runs), total=runs)
    for run in pbar:
        # Initialize model
        if gnn_model == 'GraphSAGE':
            model = models.GraphSAGE(
                num_features, num_classes, embedding_dim).to(device)
        elif gnn_model == 'APPNP':
            model = models.APPNP(
                num_features, num_classes, embedding_dim).to(device)

        # Train model on the coarsest graph
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
                logits[hierarchy_graphs[-1][0]['data'].to(device).val_mask], hierarchy_graphs[-1][0]['data'].to(device).val_y[hierarchy_graphs[-1][0]['data'].to(device).val_mask])
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                current_patience = EARLY_STOPPING_PATIENCE
            else:
                current_patience -= 1
                if current_patience == 0:
                    break
        # Evaluate
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

        # Classify
        for level in range(len(hierarchy)):
            pred_labels = classify(
                first_level_nodes=hierarchy[0],
                aggregation_method='level',
                level=level,
                num_classes=num_classes,
                num_levels=len(hierarchy),
                train_mask=train_mask,
                val_mask=val_mask,
                labels=labels,
            )
            scores[f'level{level}']['accuracy'].append(
                accuracy_score(labels[test_mask], pred_labels[test_mask]))
            scores[f'level{level}']['f1_macro'].append(
                f1_score(labels[test_mask], pred_labels[test_mask], average='macro'))
            scores[f'level{level}']['f1_micro'].append(
                f1_score(labels[test_mask], pred_labels[test_mask], average='micro'))

        for aggregation_method in AGGREGATION_METHODS:
            pred_labels = classify(
                first_level_nodes=hierarchy[0],
                aggregation_method=aggregation_method,
                level=None,
                num_classes=num_classes,
                num_levels=len(hierarchy),
                train_mask=train_mask,
                val_mask=val_mask,
                labels=labels,
            )
            scores[aggregation_method]['accuracy'].append(
                accuracy_score(labels[test_mask], pred_labels[test_mask]))
            scores[aggregation_method]['f1_macro'].append(
                f1_score(labels[test_mask], pred_labels[test_mask], average='macro'))
            scores[aggregation_method]['f1_micro'].append(
                f1_score(labels[test_mask], pred_labels[test_mask], average='micro'))

        pbar.set_description(
            f'Run {run + 1}/{runs} - F1 Macro (Post-Hoc-Learned aggregation): {np.mean(scores["post_hoc_learned_weights"]["f1_macro"]):.3f}')

    for level in range(len(hierarchy)):
        scores[f'level{level}']['accuracy_avg'] = np.mean(
            scores[f'level{level}']['accuracy'])
        scores[f'level{level}']['f1_macro_avg'] = np.mean(
            scores[f'level{level}']['f1_macro'])
        scores[f'level{level}']['f1_micro_avg'] = np.mean(
            scores[f'level{level}']['f1_micro'])

        scores[f'level{level}']['accuracy_std'] = np.std(
            scores[f'level{level}']['accuracy'])
        scores[f'level{level}']['f1_macro_std'] = np.std(
            scores[f'level{level}']['f1_macro'])
        scores[f'level{level}']['f1_micro_std'] = np.std(
            scores[f'level{level}']['f1_micro'])

    for aggregation_method in AGGREGATION_METHODS:
        scores[aggregation_method]['accuracy_avg'] = np.mean(
            scores[aggregation_method]['accuracy'])
        scores[aggregation_method]['f1_macro_avg'] = np.mean(
            scores[aggregation_method]['f1_macro'])
        scores[aggregation_method]['f1_micro_avg'] = np.mean(
            scores[aggregation_method]['f1_micro'])

        scores[aggregation_method]['accuracy_std'] = np.std(
            scores[aggregation_method]['accuracy'])
        scores[aggregation_method]['f1_macro_std'] = np.std(
            scores[aggregation_method]['f1_macro'])
        scores[aggregation_method]['f1_micro_std'] = np.std(
            scores[aggregation_method]['f1_micro'])

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--gnn_model', type=str, default='GraphSAGE')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--coarsening_method', type=str,
                        default='variation_neighborhoods')
    parser.add_argument('--reduction_ratio', type=float, default=0.5)
    parser.add_argument('--full_graph_per_level',
                        action='store_true', dest='full_graph_per_level')
    parser.add_argument('--subgraphs_per_level',
                        action='store_false', dest='full_graph_per_level')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    gnn_model = args.gnn_model
    runs = args.runs
    embedding_dim = args.embedding_dim
    coarsening_method = args.coarsening_method
    reduction_ratio = args.reduction_ratio
    full_graph_per_level = args.full_graph_per_level

    scores = dmgrl(
        dataset_name=dataset_name,
        gnn_model=gnn_model,
        runs=runs,
        embedding_dim=embedding_dim,
        coarsening_method=coarsening_method,
        reduction_ratio=reduction_ratio,
        full_graph_per_level=full_graph_per_level,
    )

    file_name = f'results/dmgrl/datasets/{args.dataset_name}/gnn_models/{args.gnn_model}/embedding_dims/{args.embedding_dim}/coarsening_methods/{args.coarsening_method}/reduction_ratios/{args.reduction_ratio}/full_graph_per_level/{args.full_graph_per_level}/scores.json'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        json.dump(scores, f)
