import torch
import torch.nn.functional as F
from torch_geometric import datasets
from torch_geometric.utils import erdos_renyi_graph, barabasi_albert_graph
from torch_geometric.data import Data


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


def random_graph(
    type,
    size,
    edge_prob=0.015,
    num_edges=2,
    num_features=500,
    num_classes=10
):
    if type == 'erdos-renyi':
        edge_index = erdos_renyi_graph(size, edge_prob)
    elif type == 'barabasi-albert':
        edge_index = barabasi_albert_graph(size, num_edges)

    x = torch.randn(size, num_features)
    y = torch.randint(0, num_classes, (size,))
    y = F.one_hot(y, num_classes).float()
    train_mask = torch.zeros(size, dtype=torch.bool)
    train_mask[:int(size * 0.8)] = 1
    val_mask = torch.zeros(size, dtype=torch.bool)
    val_mask[int(size * 0.8):int(size * 0.9)] = 1
    test_mask = torch.zeros(size, dtype=torch.bool)
    test_mask[int(size * 0.9):] = 1
    data = Data(
        x=x,
        edge_index=edge_index,
        train_y=y,
        val_y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    return data, num_classes, num_features
