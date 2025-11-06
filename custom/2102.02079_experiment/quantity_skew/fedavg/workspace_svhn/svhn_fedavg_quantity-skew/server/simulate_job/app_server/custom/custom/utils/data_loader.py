# data_loader.py - Sistema de particionamento com tensores salvos
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import fetch_openml, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ==================== CONFIGURAÇÕES ====================
DATA_DIR = "/home/marcelohaps/NVFlare-ResIA/custom/data"
SPLIT_DIR = "/home/marcelohaps/NVFlare-ResIA/custom/data/splits_partitioned"
SEED = 42

DATASET_CONFIG = {
    'mnist': {
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'normalize': ((0.1307,), (0.3081,))
    },
    'fmnist': {
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'normalize': ((0.5,), (0.5,))
    },
    'cifar10': {
        'num_classes': 10,
        'input_shape': (3, 32, 32),
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    },
    'svhn': {
        'num_classes': 10,
        'input_shape': (3, 32, 32),
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    },
    'adult': {
        'num_classes': 2,
        'input_shape': (123,),
        'type': 'tabular'
    },
    'covtype': {
        'num_classes': 7,
        'input_shape': (54,),
        'type': 'tabular'
    }
}


# ==================== FUNÇÕES AUXILIARES ====================

def get_transform(dataset_name, train=True):
    """Retorna transformações apropriadas para cada dataset"""
    config = DATASET_CONFIG.get(dataset_name)
    
    if not config or config.get('type') == 'tabular':
        return None
    
    normalize = transforms.Normalize(*config['normalize'])
    
    if dataset_name in ['mnist', 'fmnist']:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    elif dataset_name in ['cifar10', 'svhn']:
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
    
    return None


def load_full_dataset(dataset_name, train=True):
    """Carrega o dataset completo"""
    transform = get_transform(dataset_name, train=train)
    
    if dataset_name == 'cifar10':
        return datasets.CIFAR10(
            root=DATA_DIR, train=train, download=True, transform=transform
        )
    elif dataset_name == 'mnist':
        return datasets.MNIST(
            root=DATA_DIR, train=train, download=True, transform=transform
        )
    elif dataset_name == 'fmnist':
        return datasets.FashionMNIST(
            root=DATA_DIR, train=train, download=True, transform=transform
        )
    elif dataset_name == 'svhn':
        split = 'train' if train else 'test'
        return datasets.SVHN(
            root=DATA_DIR, split=split, download=True, transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} não suportado")


def load_tabular_dataset(dataset_name):    
    if dataset_name == 'adult':
        data = fetch_openml('adult', version=2, parser='auto')
        X, y = data.data, data.target
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ])
        
        X_processed = preprocessor.fit_transform(X)
        y_processed = LabelEncoder().fit_transform(y)
        
        return train_test_split(X_processed, y_processed, test_size=0.2, random_state=SEED)
    
    elif dataset_name == 'covtype':
        data = fetch_covtype()
        X, y = data.data, data.target
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y - 1, test_size=0.2, random_state=SEED)
    
    else:
        raise ValueError(f"Dataset tabular {dataset_name} não suportado")



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
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        splits = np.split(idx_k, proportions)
        
        for j, split in enumerate(splits):
            client_idx[j].extend(split.tolist())
    
    for j in range(num_clients):
        np.random.shuffle(client_idx[j])
    
    return client_idx


def partition_data_quantity_skew(dataset, num_clients, alpha=0.5, seed=42):
    np.random.seed(seed)
    
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_samples = len(labels)
    all_indices = np.arange(num_samples)
    np.random.shuffle(all_indices)
    
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    
    min_samples = 100
    proportions = proportions * (num_samples - min_samples * num_clients)
    client_sample_counts = proportions.astype(int) + min_samples
    
    diff = num_samples - client_sample_counts.sum()
    client_sample_counts[0] += diff
    
    client_idx = {}
    start_idx = 0
    
    for i in range(num_clients):
        end_idx = start_idx + client_sample_counts[i]
        client_idx[i] = all_indices[start_idx:end_idx].tolist()
        start_idx = end_idx
    
    return client_idx



def create_all_partitions(dataset_name, num_clients, alpha=0.5, seed=42, skew_type="label"):
    """
    Cria e salva todas as partições com tensores para todos os clientes.
    
    Args:
        dataset_name: Nome do dataset
        num_clients: Número de clientes
        alpha: Parâmetro de concentração Dirichlet
        seed: Seed para reprodutibilidade
        skew_type: Tipo de heterogeneidade ("label" ou "quantity")
    
    Returns:
        partition_dir: Diretório onde as partições foram salvas
    """
    print(f"\n{'='*70}")
    print(f"🔧 CRIANDO PARTIÇÕES: {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"  Clientes: {num_clients}")
    print(f"  Alpha: {alpha}")
    print(f"  Skew Type: {skew_type}")
    print(f"  Seed: {seed}\n")
    
    partition_dir = os.path.join(
        SPLIT_DIR,
        f"{dataset_name}_{skew_type}_n{num_clients}_a{alpha}_s{seed}"
    )
    os.makedirs(partition_dir, exist_ok=True)
    
    metadata_file = os.path.join(partition_dir, "metadata.pt")
    if os.path.exists(metadata_file):
        print(f"✅ Partições já existem em: {partition_dir}")
        metadata = torch.load(metadata_file)
        
        print(f"\n📊 Estatísticas das Partições:")
        for i in range(num_clients):
            stats = metadata['client_stats'][i]
            print(f"  Cliente {i}: {stats['num_samples']} amostras")
            print(f"    Classes: {stats['class_distribution']}")
        
        return partition_dir
    
    print(f"📦 Carregando dataset {dataset_name.upper()}...")
    full_dataset = load_full_dataset(dataset_name, train=True)
    
    print(f"\n🎲 Criando partição com {skew_type} skew (α={alpha})...")
    if skew_type == "quantity":
        client_idx = partition_data_quantity_skew(full_dataset, num_clients, alpha, seed)
    else:
        client_idx = partition_data_dirichlet(full_dataset, num_clients, alpha, seed)
    
    print(f"\n💾 Salvando partições...")
    metadata = {
        'dataset_name': dataset_name,
        'num_clients': num_clients,
        'alpha': alpha,
        'seed': seed,
        'skew_type': skew_type,
        'client_stats': {}
    }
    
    for client_id in range(num_clients):
        print(f"  Processando Cliente {client_id}...", end=" ")
        
        indices = client_idx[client_id]
        client_data = []
        client_labels = []
        
        for idx in indices:
            data, label = full_dataset[idx]
            client_data.append(data)
            client_labels.append(label)
        
        data_tensor = torch.stack(client_data)
        labels_tensor = torch.tensor(client_labels, dtype=torch.long)
        
        client_file = os.path.join(partition_dir, f"client_{client_id}.pt")
        torch.save({
            'data': data_tensor,
            'labels': labels_tensor,
            'indices': indices
        }, client_file)
        
        unique, counts = np.unique(client_labels, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        
        metadata['client_stats'][client_id] = {
            'num_samples': len(indices),
            'class_distribution': class_dist
        }
        
        print(f"✓ ({len(indices)} amostras)")
    
    torch.save(metadata, metadata_file)
    
    print(f"\n{'='*70}")
    print(f"✅ PARTIÇÕES CRIADAS COM SUCESSO!")
    print(f"{'='*70}")
    print(f"📁 Diretório: {partition_dir}")
    print(f"\n📊 Resumo das Partições:")
    
    for i in range(num_clients):
        stats = metadata['client_stats'][i]
        print(f"  Cliente {i}: {stats['num_samples']} amostras")
        print(f"    Classes: {stats['class_distribution']}")
    
    print(f"\n{'='*70}\n")
    
    return partition_dir


def load_client_partition(dataset_name, client_id, num_clients, alpha=0.5, seed=42, 
                          skew_type="label", batch_size=64, shuffle=True):
    """
    Carrega partição de um cliente específico.
    
    Args:
        dataset_name: Nome do dataset
        client_id: ID do cliente
        num_clients: Número total de clientes
        alpha: Parâmetro de concentração Dirichlet
        seed: Seed para reprodutibilidade
        skew_type: Tipo de heterogeneidade
        batch_size: Tamanho do batch
        shuffle: Se deve embaralhar os dados
    
    Returns:
        DataLoader com os dados do cliente
    """
    partition_dir = os.path.join(
        SPLIT_DIR,
        f"{dataset_name}_{skew_type}_n{num_clients}_a{alpha}_s{seed}"
    )
    
    client_file = os.path.join(partition_dir, f"client_{client_id}.pt")
    
    if not os.path.exists(client_file):
        raise FileNotFoundError(
            f"❌ Partição não encontrada: {client_file}\n"
            f"Execute create_all_partitions() primeiro!"
        )
    
    client_data = torch.load(client_file)
    
    dataset = TensorDataset(client_data['data'], client_data['labels'])
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )
    
    return loader


def load_test_data(dataset_name, batch_size=64):
    test_dataset = load_full_dataset(dataset_name, train=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader



def get_client_id_from_site_name(site_name):
    if site_name.startswith('site-'):
        return int(site_name.split('-')[1]) - 1
    else:
        try:
            return int(site_name)
        except:
            return 0


def load_train_data(site_name, dataset_name='cifar10', batch_size=64, 
                    num_clients=10, alpha=0.5, skew_type="label"):
    """
    Função de compatibilidade para carregar dados de treinamento.
    
    Args:
        site_name: Nome do site (ex: 'site-1')
        dataset_name: Nome do dataset
        batch_size: Tamanho do batch
        num_clients: Número de clientes
        alpha: Parâmetro alpha
        skew_type: Tipo de skew
    
    Returns:
        DataLoader com dados do cliente
    """
    client_id = get_client_id_from_site_name(site_name)
    
    return load_client_partition(
        dataset_name=dataset_name,
        client_id=client_id,
        num_clients=num_clients,
        alpha=alpha,
        seed=SEED,
        skew_type=skew_type,
        batch_size=batch_size,
        shuffle=True
    )


def load_val_data(site_name, dataset_name='cifar10', batch_size=64):
    """
    Função de compatibilidade para carregar dados de validação.
    
    Args:
        site_name: Nome do site
        dataset_name: Nome do dataset
        batch_size: Tamanho do batch
        num_clients: Número de clientes (não usado, mantido para compatibilidade)
    
    Returns:
        DataLoader com dados de teste
    """
    return load_test_data(dataset_name, batch_size)



if __name__ == "__main__":
    print("Exemplo de uso do novo data_loader.py\n")
    
    # 1. Criar partições para CIFAR10
    partition_dir = create_all_partitions(
        dataset_name='cifar10',
        num_clients=10,
        alpha=0.5,
        seed=42,
        skew_type='label'
    )
    
    # 2. Carregar dados de um cliente
    print("Testando carregamento de dados do Cliente 0...")
    train_loader = load_client_partition(
        dataset_name='cifar10',
        client_id=0,
        num_clients=10,
        alpha=0.5,
        batch_size=32
    )
    
    # 3. Testar um batch
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch[0].shape}")
    print(f"Labels shape: {batch[1].shape}")
    print("\n✅ Tudo funcionando corretamente!")
