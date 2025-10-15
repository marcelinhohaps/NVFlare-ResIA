import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SPLIT_DIR = "/home/marcelohaps/NVFlare-ResIA/custom/data/splits_dirichlet"
ALPHA = 0.3  #parâmetro de concentração Dirichlet (menor = mais heterogêneo)
NUM_CLASSES = 10
SEED = 42


def partition_data_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    np.random.seed(seed)
    
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    client_idx = {i: [] for i in range(num_clients)}
    
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        proportions = np.array([
            p * (len(client_idx[j]) < num_samples / num_clients) 
            for j, p in enumerate(proportions)
        ])
        proportions = proportions / proportions.sum()
        
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)
        
        for j, split in enumerate(splits):
            client_idx[j].extend(split.tolist())
    
    for j in range(num_clients):
        np.random.shuffle(client_idx[j])
    
    return client_idx


def get_or_create_partition(num_clients, alpha=0.5, seed=42):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    
    partition_file = os.path.join(
        SPLIT_DIR, 
        f"partition_n{num_clients}_a{alpha}_s{seed}.npy"
    )
    
    if os.path.exists(partition_file):
        client_idx = np.load(partition_file, allow_pickle=True).item()
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_dataset = datasets.CIFAR10(
            root='data/CIFAR10', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        client_idx = partition_data_dirichlet(full_dataset, num_clients, alpha, seed)
        
        np.save(partition_file, client_idx)
        print(f"💾 Partição salva em: {partition_file}")
        
        for i in range(num_clients):
            labels_i = np.array([full_dataset.targets[j] for j in client_idx[i]])
            unique, counts = np.unique(labels_i, return_counts=True)
            print(f"  Cliente {i}: {len(client_idx[i])} amostras - "
                  f"Classes: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    return client_idx


def get_client_id_from_site_name(site_name):
    if site_name.startswith('site-'):
        return int(site_name.split('-')[1]) - 1
    else:
        try:
            return int(site_name)
        except:
            return 0


def load_train_data(site_name, batch_size=32, num_clients=4, alpha=ALPHA):
    client_id = get_client_id_from_site_name(site_name)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = datasets.CIFAR10(
        root='data/',
        train=True,
        download=True,
        transform=transform_train
    )
    
    client_idx = get_or_create_partition(num_clients, alpha, SEED)
    
    indices = client_idx[client_id]
    client_dataset = Subset(full_dataset, indices)
    
    train_loader = DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    return train_loader


def load_val_data(site_name, batch_size=32, num_clients=4):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = datasets.CIFAR10(
        root='/tmp/data',
        train=False,
        download=True,
        transform=transform_test
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return val_loader
