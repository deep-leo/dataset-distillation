import random
import torch
import torch.optim as optim

class Client:
    """Class representing a physical device (such as a smartphone) with a
    quarantined dataset.
    """
    def __init__(self, name, net, criterion, dataloader, selection_prob):
        self.name = name
        self.net = net
        self.optimizer = optim.Adam(net.parameters())
        self.criterion = criterion
        self.dataloader = dataloader
        self.selection_prob = selection_prob

        self.size = len(dataloader.dataset)
        # assumes entire neural network is on a single device
        self.device = next(net.parameters()).device

    def selected(self):
        """Determes randomly if a client is choosen for training.
        """
        return random.random() < self.selection_prob

    def train(self, epochs):
        """Trains network for a given number of epochs.
        """
        print('Training client', self.name)
        self.net.train()

        for e in range(1, epochs + 1):
            epoch_loss = 0
            iters = 0
            for i, (images, labels) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                out = self.net(images.to(self.device))
                loss = self.criterion(out, labels.to(self.device))
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                iters += 1
            print(f'Epoch: {e} Loss: {epoch_loss / iters:.4f}')

        return epoch_loss    


class Server:
    """Centralized server connected to all clients, capable of aggregating and
    broadcasting parameters.
    """

    def __init__(self, net, clients, testloader, classes, notebook):
        self.net = net
        self.clients = clients
        self.testloader = testloader
        self.classes = classes
        self.notebook = notebook

        self.size = sum([client.size for client in clients])
        # assumes neural network is on a single device
        self.device = next(net.parameters()).device
        # ensure client nets have same init as server's net
        for client in clients:
            client.net.load_state_dict(self.net.state_dict())

    def average(self, trained_clients):
        """Updates server parameters by averaging client networks.
        """
        new_state_dict = self.net.state_dict()
        trained_size = sum(client.size for client in trained_clients)

        for key, param in new_state_dict.items():
            average = torch.zeros_like(param)
            for client in trained_clients:
                average += client.size / trained_size * client.net.state_dict()[key]
            prop_trained = trained_size / self.size
            new_state_dict[key] = prop_trained * average + (1 - prop_trained) * param

        self.net.load_state_dict(new_state_dict)

    def evaluate(self):
        """Measures performance of global model on a test set.
        """
        print('Evaluating server model')
        self.net.eval()

        class_correct = len(self.classes) * [0]
        class_total = len(self.classes) * [0]
        for images, labels in self.testloader:
            outputs = self.net(images.to(self.device))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels.to(self.device)).squeeze()
            for i in range(self.testloader.batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        accuracy = 100 * sum(class_correct) / sum(class_total)
        print(f'Accuracy: {accuracy:.2f}%')

        accuracies = {'all': accuracy}
        for i, clazz in enumerate(self.classes):
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'Class: {clazz} Accuracy: {accuracy:.2f}%')
            accuracies[clazz] = accuracy

        return accuracies

    def train(self, rounds, epochs_per_round):
        """Initiates federated learning session.
        """
        for r in range(1, rounds + 1):
            print('Communication round', r)
            # get available clients
            participants = [client for client in self.clients if client.selected()]

            losses = {}
            for client in participants:
                # sync client model with server model before training
                client.net.load_state_dict(self.net.state_dict())
                # record loss in dictionary
                losses[client.name] = client.train(epochs_per_round)
            self.notebook.add_scalars('Loss', losses, global_step=r)

            if len(participants) > 0:
                self.average(participants)

            accuracies = self.evaluate()
            self.notebook.add_scalars('Accuracy', accuracies, global_step=r)

        self.notebook.lineplot('Accuracy_all', save=True)
        self.notebook.relplot('Accuracy', save=True)
        self.notebook.relplot('Loss', save=True)
        self.notebook.save_tar()

 