import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import models.models as models
from utils.datasets import load_dataset

EARLY_STOPPING_PATIENCE = 5


def baseline(
    dataset_name,
    gnn_model,
    runs,
    embedding_dim,
):
    print(
        f'Baseline - Dataset: {dataset_name}, GNN Model: {gnn_model}, Embedding Dim: {embedding_dim}')

    # Load dataset
    data, num_classes, num_features = load_dataset(dataset_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    scores = {
        'accuracy': [],
        'f1_macro': [],
        'f1_micro': [],
    }
    pbar = tqdm(range(runs), total=runs)
    for run in pbar:
        # Initialize model
        if gnn_model == 'GraphSAGE':
            model = models.GraphSAGE(
                num_features, num_classes, embedding_dim).to(device)
        elif gnn_model == 'APPNP':
            model = models.APPNP(
                num_features, num_classes, embedding_dim).to(device)

        optimizer = torch.optim.RMSprop(model.parameters())

        # Train model
        current_patience = EARLY_STOPPING_PATIENCE
        best_val_loss = None
        best_model = None
        model.train()
        while True:
            optimizer.zero_grad()
            _, logits = model(data)
            loss = F.cross_entropy(
                logits[data.train_mask], data.train_y[data.train_mask])
            loss.backward()
            optimizer.step()
            val_loss = F.cross_entropy(
                logits[data.val_mask], data.val_y[data.val_mask])
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                current_patience = EARLY_STOPPING_PATIENCE
            else:
                current_patience -= 1
                if current_patience == 0:
                    break
        model.load_state_dict(best_model)
        model.eval()
        _, logits = model(data)
        pred_labels = logits.argmax(dim=1).detach().cpu().numpy()
        true_labesl = data.train_y.argmax(dim=1).detach().cpu().numpy()
        test_mask = data.test_mask.detach().cpu().numpy()
        scores['accuracy'].append(accuracy_score(
            true_labesl[test_mask], pred_labels[test_mask]))
        scores['f1_macro'].append(f1_score(
            true_labesl[test_mask], pred_labels[test_mask], average='macro'))
        scores['f1_micro'].append(f1_score(
            true_labesl[test_mask], pred_labels[test_mask], average='micro'))
        pbar.set_description(
            f'Run {run + 1}/{runs} - F1 Macro: {np.mean(scores["f1_macro"]):.3f}')

    scores['accuracy_std'] = np.std(scores['accuracy'])
    scores['f1_macro_std'] = np.std(scores['f1_macro'])
    scores['f1_micro_std'] = np.std(scores['f1_micro'])

    scores['accuracy_avg'] = sum(scores['accuracy']) / runs
    scores['f1_macro_avg'] = sum(scores['f1_macro']) / runs
    scores['f1_micro_avg'] = sum(scores['f1_micro']) / runs

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--gnn_model', type=str, default='GraphSAGE')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=128)
    args = parser.parse_args()

    scores = baseline(
        args.dataset_name,
        args.gnn_model,
        args.runs,
        args.embedding_dim,
    )
    file_name = f'results/baseline/datasets/{args.dataset_name}/gnn_models/{args.gnn_model}/embedding_dims/{args.embedding_dim}/scores.json'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        json.dump(scores, f)
