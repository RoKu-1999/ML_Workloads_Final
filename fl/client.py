import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import sys
import time
# Start client
import os
import logging


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net1(nn.Module):
    def __init__(self) -> None:
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net2(nn.Module):
    def __init__(self) -> None:
        super(Net2, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # Define dropout
        self.dropout = nn.Dropout(0.5)
        # We will determine the size of the linear layers automatically.
        # Placeholder for the number of features after conv layers
        self._to_linear = None
        # Define the fully connected layers
        self.fc1 = nn.Linear(self.to_linear, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 84)
        self.fc4 = nn.Linear(84, 10)

    def to_linear(self, x):
        if self._to_linear is None:
            # Apply conv and pool layers to dummy variable
            self._to_linear = self.pool(F.relu(self.conv3(self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(x)))))))))
            # Flatten the output to get the total number of features
            self._to_linear = int(np.prod(self._to_linear.size()[1:]))
        return self._to_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolution and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the tensor with the correct size using the to_linear function
        x = x.view(-1, self.to_linear(x))
        # Apply fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        x = self.fc4(x)
        return x


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        logging.info(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def load_datasets(num_clients: int, batch_size : int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=False, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=False, transform=transform)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // 128 #// num_clients
    lengths = [partition_size] * 128 #num_clients

    i = 0
    while (sum(lengths)!=len(trainset)):
        lengths[i] = lengths[i] + 1
        i = (i + 1) % 128 #num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        logging.info(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        logging.info(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        start_time = time.perf_counter()
        train(self.net, self.trainloader, epochs=50)
        end_time = time.perf_counter()
        f.write(str(end_time-start_time)+',')
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        logging.info(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        start_time = time.perf_counter()
        loss, accuracy = test(self.net, self.valloader)
        end_time = time.perf_counter()
        f.write(str(end_time-start_time)+',')
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid, trainloaders, valloaders) -> FlowerClient:
    net = Net1().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


# Start Flower client
if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    NUM_CLIENTS = int(sys.argv[2])
    client_id = int(sys.argv[3])
    secure = int(sys.argv[4])
    epochs= int(sys.argv[5])
    sgx= int(sys.argv[6])
    net = Net1().to(DEVICE)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS, batch_size)
    client = client_fn(client_id, trainloaders, valloaders)
    f = open("results/client_result_"+str(epochs)+".txt", "a")
    f.write(str(client_id)+',')
    f.write(str(epochs)+',')
    f.write(str(NUM_CLIENTS)+',')
    f.write(str(batch_size)+',')
    f.write(str(sgx)+',')
    f.write(str(secure)+',')
    f.write(str(total_params)+',')
    start_time = time.perf_counter()
    if secure == 0:
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=client,
        )
    else:
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=client,
            root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
        )
    end_time = time.perf_counter()
    f.write(str(end_time-start_time)+"\n")
    f.close()
