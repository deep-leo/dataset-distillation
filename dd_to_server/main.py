import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from basics import task_loss, final_objective_loss, evaluate_steps
from models import LeNet




'''
Load dataset and return trainset, testset for server

'''
def load_dataset(dataset_name, data_dir):
    """Downloads `dataset_name` and stores it on disk.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if dataset_name == 'mnist':
        # 60_000 training, 10_000 test
        trainset = torchvision.datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    elif dataset_name == 'cifar10':
        # 50_000 training, 10_000 test
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise ValueError(f'Expected mnist, cifar10. Got {dataset_name}.')
    return trainset, testset, classes




'''
Create clients and servers

'''
def create_clients(server_net, dataloaders, availability):
    """Creates a list of clients from the dataset returned by `_partition`.
    Each local model is a deepcopy of `server_net`.
    """
    clients = []
    for i, dataloader in enumerate(dataloaders):
        client = Client(str(i), copy.deepcopy(server_net), nn.CrossEntropyLoss(), dataloader, availability)
        clients.append(client)
    return clients


def create_server(net, clients, testset, classes):
    """Constructs a Server object and connects it with the clients.
    """
    return Server(net, clients, torchdata.DataLoader(testset, batch_size=4), classes, Notebook(''))



'''
Initialize a server, broadcast its initial weights to all clients

'''
def init_weights(net, state):
    init_type, init_param = state.init, state.init_param

    if init_type == 'imagenet_pretrained':
        assert net.__class__.__name__ == 'LeNet'
        state_dict = torchvision.models.alexnet(pretrained=True).state_dict()
        state_dict['classifier.6.weight'] = torch.zeros_like(net.classifier[6].weight)
        state_dict['classifier.6.bias'] = torch.ones_like(net.classifier[6].bias)
        net.load_state_dict(state_dict)
        del state_dict
        return net

    def init_func(m):
        classname = m.__class__.__name__
        if classname.startswith('Conv') or classname == 'Linear':
            if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias, 0.0)
            if getattr(m, 'weight', None) is not None:
                if init_type == 'normal':
                    init.normal_(m.weight, 0.0, init_param)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=init_param)
                elif init_type == 'xavier_unif':
                    init.xavier_uniform_(m.weight, gain=init_param)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
                elif init_type == 'kaiming_out':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=init_param)
                elif init_type == 'default':
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif 'Norm' in classname:
            if getattr(m, 'weight', None) is not None:
                m.weight.data.fill_(1)
            if getattr(m, 'bias', None) is not None:
                m.bias.data.zero_()

    net.apply(init_func)
    return net




'''
Client receive server weights, load model, split dataset and learn
'''
def load_train_models():
    if state.train_nets_type == 'unknown_init':
        model, = networks.get_networks(state, N=1)
        return [model for _ in range(state.local_n_nets)]
    elif state.train_nets_type == 'known_init':
        return networks.get_networks(state, N=state.local_n_nets)
    elif state.train_nets_type == 'loaded':
        models = networks.get_networks(state, N=state.local_n_nets)
        with state.pretend(phase='train'):  # in case test_nets_type == same_as_train
            model_dir = state.get_model_dir()
        start_idx = state.world_rank * state.local_n_nets
        for n, model in enumerate(models, start_idx):
            model_path = os.path.join(model_dir, 'net_{:04d}'.format(n))
            model.load_state_dict(torch.load(model_path, map_location=state.device))
        logging.info('Loaded checkpoints [{} ... {}) from {}'.format(
            start_idx, start_idx + state.local_n_nets, model_dir))
        return models
    else:
        raise ValueError("train_nets_type: {}".format(state.train_nets_type))



# Use class split, each class has only 1 corresponding client
def class_split_data(dataset, shard_size, lengths, batch_size):
    if len(dataset) % shard_size != 0:
        raise ValueError('Shard size does not divide the length of the dataset.')

    if any([length % shard_size != 0 for length in lengths]):
        raise ValueError('Lengths are not divisible by shard size.')

    if sum(lengths) != len(dataset):
        raise ValueError('Sum of lengths does not equal the length of the dataset.')

    # sort dataset indices by their label
    index = []
    for i, (_, label) in enumerate(dataset):
        index.append((i, label))
    index = sorted(index, key=lambda t: t[1])
    index = [i for i, label in index]

    # divide dataset into shards
    shards = [index[i:i + shard_size] for i in range(0, len(dataset), shard_size)]

    # group shards into subsets
    subsets = []
    for length in lengths:
        shards_sample = random.sample(shards, length // shard_size)
        for shard in shards_sample:
            shards.remove(shard)
        # sum concatenates lists
        subset = torch.utils.data.Subset(dataset, sum(shards_sample, []))
        subsets.append(subset)

    trainloaders = [DataLoader(dataset=subset, batch_size=batch_size, shuffle=True)
                    for subset in subsets]
    return trainloaders


def load_model(model_name, image, output_size):
    return LeNet(*image.size(), output_size)



'''
Step 3
Clients generate distilled images wrt server's weight

'''



'''
Step 4
Clients send distilled images to server

'''


def train(state, model, epoch, optimizer):
    model.train()
    for it, (data, target) in enumerate(state.train_loader):
        data, target = data.to(state.device, non_blocking=True), target.to(state.device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if state.log_interval > 0 and it % state.log_interval == 0:
            log_str = 'Epoch: {:4d} ({:2.0f}%)\tTrain Loss: {: >7.4f}'.format(
                epoch, 100. * it / len(state.train_loader), loss.item())
            if it == 0 or (state.log_interval > 0 and it % state.log_interval == 0):
                acc, loss = evaluate_models(state, [model])
                log_str += '\tTest Acc: {: >5.2f}%\tTest Loss: {: >7.4f}'.format(acc.item() * 100, loss.item())
                model.train()
            logging.info(log_str)



def main(cfg):

# load dataset, get test and train
    trainset, testset, classes = load_dataset(cfg.dataset, hydra.utils.to_absolute_path('data'))
    image, _ = next(iter(testset))
    net = load_model(cfg.model, image, len(classes)).to(device)

    lengths = cfg.clients * [len(trainset) // cfg.clients]
    shard_size = len(trainset) // cfg.clients // cfg.shards_in_subset

    trainloaders = non_iid_partition(trainset, shard_size, lengths, cfg.batch_size)


# create clients and server
    clients = create_clients(net, trainloaders, cfg.selection_prob)
    server = create_server(net, clients, testset, classes)




#  Initialize a server, broadcast its initial weights to all clients
    model_dir = state.get_model_dir()
    networks.get_networks(state, N=state.local_n_nets)


    steps = train_distilled_image.distill(state, state.models)
    evaluate_steps(state, steps,
                   'distilled with {} steps and {} epochs'.format(state.distill_steps, state.distill_epochs),
                   test_all=True)


if __name__ == '__main__':





