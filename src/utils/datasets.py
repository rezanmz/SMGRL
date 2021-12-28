import torch
import torch.nn.functional as F
from torch_geometric import datasets


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_mask(dataset):
    indices = []
    for i in range(dataset.num_classes):
        index = (dataset.data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    train_index = torch.cat([i[:int(0.2 * i.shape[0])]
                            for i in indices], dim=0)
    val_index = torch.cat(
        [i[int(0.2 * i.shape[0]):int(0.4 * i.shape[0])] for i in indices], dim=0)
    test_index = torch.cat([i[int(0.4 * i.shape[0]):] for i in indices], dim=0)
    dataset.data.train_mask = index_to_mask(
        train_index, size=dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(
        val_index, size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(
        test_index, size=dataset.data.num_nodes)
    return dataset


def load_dataset(name):
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = datasets.Planetoid(root='data/planetoid', name=name)
    if name == 'actor':
        dataset = datasets.Actor(root='data/actor')
        del dataset.data.train_mask
        del dataset.data.val_mask
        del dataset.data.test_mask
    if name == 'flickr':
        dataset = datasets.Flickr('data/flickr')
    if name == 'reddit':
        dataset = datasets.Reddit('data/reddit')
    if name == 'reddit2':
        dataset = datasets.Reddit2('data/reddit2')
    if name == 'amazon-products':
        dataset = datasets.AmazonProducts('data/amazon-prodoucts')
    if name in ['amazon-computers', 'amazon-photo']:
        dataset = datasets.Amazon(
            root='data/amazon', name='Computers' if name == 'amazon-computers' else 'Photo')
    if name == 'dblp':
        dataset = datasets.CitationFull('data/dblp', 'dblp')
    if 'train_mask' not in dataset.data:
        dataset = random_mask(dataset)
    data, num_classes, num_features = dataset.data, dataset.num_classes, dataset.num_features
    if name != 'amazon-products':
        data.y = F.one_hot(data.y, num_classes).float()
    data.train_y, data.val_y = data.y.detach().clone(), data.y.detach().clone()
    del data.y
    return data, num_classes, num_features
