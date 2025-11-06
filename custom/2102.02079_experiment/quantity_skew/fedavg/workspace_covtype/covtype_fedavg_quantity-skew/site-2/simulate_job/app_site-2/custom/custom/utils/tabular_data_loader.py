# data_loader_tabular.py - Sistema de particionamento para datasets tabulares
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
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


def load_tabular_dataset(dataset_name):
    """
    Carrega e processa dataset tabular.
    
    Args:
        dataset_name: Nome do dataset ('adult' ou 'covtype')
    
    Returns:
        X_train, X_test, y_train, y_test: Arrays numpy processados
    """
    if dataset_name == 'adult':
        print(f"📦 Baixando dataset Adult...")
        data = fetch_openml('adult', version=2, parser='auto')
        X, y = data.data, data.target
        
        print(f"  Processando features categóricas e numéricas...")
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ])
        
        X_processed = preprocessor.fit_transform(X)
        y_processed = LabelEncoder().fit_transform(y)
        
        print(f"  Shape final: {X_processed.shape}, Classes: {len(np.unique(y_processed))}")
        
        return train_test_split(X_processed, y_processed, test_size=0.2, random_state=SEED)
    
    elif dataset_name == 'covtype':
        print(f"📦 Baixando dataset Covertype...")
        data = fetch_covtype()
        X, y = data.data, data.target
        
        print(f"  Aplicando StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Ajustar labels para começar em 0 (originalmente começa em 1)
        y_adjusted = y - 1
        
        print(f"  Shape final: {X_scaled.shape}, Classes: {len(np.unique(y_adjusted))}")
        
        return train_test_split(X_scaled, y_adjusted, test_size=0.2, random_state=SEED)
    
    else:
        raise ValueError(f"Dataset tabular {dataset_name} não suportado. "
                        f"Datasets disponíveis: {list(DATASET_CONFIG.keys())}")


def partition_tabular_dirichlet(X, y, num_clients, alpha=0.5, seed=42):
    """
    Particiona dados tabulares usando Dirichlet distribution (label skew).
    
    Args:
        X: Array numpy com features (n_samples, n_features)
        y: Array numpy com labels (n_samples,)
        num_clients: Número de clientes
        alpha: Parâmetro de concentração Dirichlet (menor = mais heterogêneo)
        seed: Seed para reprodutibilidade
    
    Returns:
        client_idx: Dicionário com índices para cada cliente {client_id: [indices]}
    """
    np.random.seed(seed)
    
    labels = np.array(y)
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    print(f"  Distribuindo {num_samples} amostras em {num_clients} clientes")
    print(f"  Classes: {num_classes}, Alpha: {alpha}")
    
    client_idx = {i: [] for i in range(num_clients)}
    
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Distribuição Dirichlet para a classe k
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        
        # Calcular pontos de divisão cumulativos
        cumsum_proportions = np.cumsum(proportions) * len(idx_k)
        split_points = cumsum_proportions.astype(int)[:-1]
        
        # Dividir índices da classe k entre os clientes
        splits = np.split(idx_k, split_points)
        
        for j, split in enumerate(splits):
            client_idx[j].extend(split.tolist())
    
    # Embaralhar índices de cada cliente
    for j in range(num_clients):
        np.random.shuffle(client_idx[j])
    
    return client_idx


def partition_tabular_quantity_skew(X, y, num_clients, alpha=0.5, seed=42):
    """
    Particiona dados tabulares usando quantity skew (heterogeneidade de quantidade).
    
    Args:
        X: Array numpy com features (n_samples, n_features)
        y: Array numpy com labels (n_samples,)
        num_clients: Número de clientes
        alpha: Parâmetro de concentração Dirichlet (menor = mais heterogêneo)
        seed: Seed para reprodutibilidade
    
    Returns:
        client_idx: Dicionário com índices para cada cliente {client_id: [indices]}
    """
    np.random.seed(seed)
    
    num_samples = len(y)
    all_indices = np.arange(num_samples)
    np.random.shuffle(all_indices)
    
    # Gerar proporções usando Dirichlet
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    
    # Garantir mínimo de amostras por cliente
    min_samples = 100
    proportions = proportions * (num_samples - min_samples * num_clients)
    client_sample_counts = proportions.astype(int) + min_samples
    
    # Ajustar diferença de arredondamento
    diff = num_samples - client_sample_counts.sum()
    client_sample_counts[0] += diff
    
    print(f"  Distribuindo {num_samples} amostras em {num_clients} clientes")
    print(f"  Alpha: {alpha}, Mínimo por cliente: {min_samples}")
    
    # Distribuir índices sequencialmente
    client_idx = {}
    start_idx = 0
    
    for i in range(num_clients):
        end_idx = start_idx + client_sample_counts[i]
        client_idx[i] = all_indices[start_idx:end_idx].tolist()
        start_idx = end_idx
    
    return client_idx


def create_all_partitions(dataset_name, num_clients, alpha=0.5, seed=42, skew_type="quantity"):
    """
    Cria e salva todas as partições com tensores para todos os clientes.
    
    Args:
        dataset_name: Nome do dataset ('adult' ou 'covtype')
        num_clients: Número de clientes
        alpha: Parâmetro de concentração Dirichlet
        seed: Seed para reprodutibilidade
        skew_type: Tipo de heterogeneidade ("label" ou "quantity")
    
    Returns:
        partition_dir: Diretório onde as partições foram salvas
    """
    print(f"\n{'='*70}")
    print(f"🔧 CRIANDO PARTIÇÕES TABULARES: {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"  Clientes: {num_clients}")
    print(f"  Alpha: {alpha}")
    print(f"  Skew Type: {skew_type}")
    print(f"  Seed: {seed}\n")
    
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset {dataset_name} não suportado. "
                        f"Datasets disponíveis: {list(DATASET_CONFIG.keys())}")
    
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
    
    # Carregar dataset
    X_train, X_test, y_train, y_test = load_tabular_dataset(dataset_name)
    
    # Converter para tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    print(f"\n🎲 Criando partição com {skew_type} skew (α={alpha})...")
    if skew_type == "quantity":
        client_idx = partition_tabular_quantity_skew(
            X_train, y_train, num_clients, alpha, seed
        )
    else:
        client_idx = partition_tabular_dirichlet(
            X_train, y_train, num_clients, alpha, seed
        )
    
    print(f"\n💾 Salvando partições...")
    metadata = {
        'dataset_name': dataset_name,
        'num_clients': num_clients,
        'alpha': alpha,
        'seed': seed,
        'skew_type': skew_type,
        'client_stats': {},
        'is_tabular': True,
        'input_shape': DATASET_CONFIG[dataset_name]['input_shape'],
        'num_classes': DATASET_CONFIG[dataset_name]['num_classes']
    }
    
    # Salvar partições de cada cliente
    for client_id in range(num_clients):
        print(f"  Processando Cliente {client_id}...", end=" ")
        
        indices = client_idx[client_id]
        client_data = X_train_tensor[indices]
        client_labels = y_train_tensor[indices]
        
        client_file = os.path.join(partition_dir, f"client_{client_id}.pt")
        torch.save({
            'data': client_data,
            'labels': client_labels,
            'indices': indices
        }, client_file)
        
        unique, counts = np.unique(client_labels.numpy(), return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        
        metadata['client_stats'][client_id] = {
            'num_samples': len(indices),
            'class_distribution': class_dist
        }
        
        print(f"✓ ({len(indices)} amostras)")
    
    # Salvar dados de teste
    test_file = os.path.join(partition_dir, "test.pt")
    torch.save({
        'data': X_test_tensor,
        'labels': y_test_tensor
    }, test_file)
    print(f"  Dados de teste salvos: {len(y_test_tensor)} amostras")
    
    # Salvar metadata
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


def load_test_data(dataset_name, num_clients=10, alpha=0.5, seed=42, 
                   skew_type="label", batch_size=64):
    """
    Carrega dados de teste.
    
    Args:
        dataset_name: Nome do dataset
        num_clients: Número de clientes (para encontrar o diretório correto)
        alpha: Parâmetro alpha
        seed: Seed
        skew_type: Tipo de skew
        batch_size: Tamanho do batch
    
    Returns:
        DataLoader com dados de teste
    """
    partition_dir = os.path.join(
        SPLIT_DIR,
        f"{dataset_name}_{skew_type}_n{num_clients}_a{alpha}_s{seed}"
    )
    
    test_file = os.path.join(partition_dir, "test.pt")
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(
            f"❌ Dados de teste não encontrados: {test_file}\n"
            f"Execute create_all_partitions() primeiro!"
        )
    
    test_data = torch.load(test_file)
    
    dataset = TensorDataset(test_data['data'], test_data['labels'])
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return loader


def get_client_id_from_site_name(site_name):
    """
    Extrai o ID do cliente a partir do nome do site.
    
    Args:
        site_name: Nome do site (ex: 'site-1' ou '1')
    
    Returns:
        client_id: ID do cliente (0-indexed)
    """
    if site_name.startswith('site-'):
        return int(site_name.split('-')[1]) - 1
    else:
        try:
            return int(site_name)
        except:
            return 0


def load_train_data(site_name, dataset_name='covtype', batch_size=64,
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


def load_val_data(site_name, dataset_name='covtype', batch_size=64,
                  num_clients=10, alpha=0.5, skew_type="label"):
    """
    Função de compatibilidade para carregar dados de validação.
    
    Args:
        site_name: Nome do site
        dataset_name: Nome do dataset
        batch_size: Tamanho do batch
        num_clients: Número de clientes
        alpha: Parâmetro alpha
        skew_type: Tipo de skew
    
    Returns:
        DataLoader com dados de teste
    """
    return load_test_data(
        dataset_name=dataset_name,
        num_clients=num_clients,
        alpha=alpha,
        seed=SEED,
        skew_type=skew_type,
        batch_size=batch_size
    )


if __name__ == "__main__":
    print("Exemplo de uso do data_loader_tabular.py\n")
    
    # 1. Criar partições para covtype
    print("=" * 70)
    print("EXEMPLO 1: Criando partições para covtype")
    print("=" * 70)
    partition_dir = create_all_partitions(
        dataset_name='covtype',
        num_clients=10,
        alpha=0.5,
        seed=42,
        skew_type='label'
    )
    
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Carregando dados do Cliente 0")
    print("=" * 70)
    train_loader = load_client_partition(
        dataset_name='covtype',
        client_id=0,
        num_clients=10,
        alpha=0.5,
        batch_size=32
    )
    
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch[0].shape}")
    print(f"Labels shape: {batch[1].shape}")
    
    # 4. Carregar dados de teste
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Carregando dados de teste")
    print("=" * 70)
    test_loader = load_test_data(
        dataset_name='covtype',
        num_clients=10,
        alpha=0.5,
        batch_size=64
    )
    
    test_batch = next(iter(test_loader))
    print(f"Test batch shape: {test_batch[0].shape}")
    print(f"Test labels shape: {test_batch[1].shape}")
    
    print("\n✅ Tudo funcionando corretamente!")