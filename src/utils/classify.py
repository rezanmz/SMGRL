import numpy as np
import torch
import torch.nn.functional as F
from baseline import EARLY_STOPPING_PATIENCE
from models import models
from .hierarchy import aggregate


def classify(
    first_level_nodes,
    aggregation_method,
    level,
    num_classes,
    num_levels,
    train_mask,
    val_mask,
    labels
):
    embeddings = []
    for node in first_level_nodes:
        if aggregation_method == 'mean':
            embeddings.append(np.mean(aggregate(node), axis=0))
        elif aggregation_method == 'concat':
            embeddings.append(np.concatenate(aggregate(node)))
        elif aggregation_method == 'post_hoc_learned_weights':
            embeddings.append(aggregate(node))
        elif aggregation_method == 'level':
            embeddings.append(aggregate(node)[level])
    embeddings = np.array(embeddings)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if aggregation_method == 'post_hoc_learned_weights':
        classifier = models.PostHocLearnedWeightsClassifier(
            embeddings.shape[2],
            num_classes,
            num_levels,
        ).to(device)
    else:
        classifier = models.Classifier(
            embeddings.shape[1], num_classes).to(device)
    optimizer = torch.optim.RMSprop(classifier.parameters())
    x = torch.FloatTensor(embeddings).to(device)
    train_mask = torch.BoolTensor(train_mask).to(device)
    val_mask = torch.BoolTensor(val_mask).to(device)
    labels = torch.LongTensor(labels).to(device)
    classifier.train()
    best_val = None
    best_model = None
    patience = EARLY_STOPPING_PATIENCE
    while True:
        optimizer.zero_grad()
        out = classifier(x)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        val_loss = F.cross_entropy(out[val_mask], labels[val_mask])
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_model = classifier.state_dict()
            patience = EARLY_STOPPING_PATIENCE
        else:
            patience -= 1
            if patience == 0:
                break
    classifier.load_state_dict(best_model)
    classifier.eval()
    pred_labels = classifier(x).argmax(1).cpu().detach().numpy()
    return pred_labels
