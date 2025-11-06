#!/usr/bin/env python3
"""
Script para validar todos os modelos globais salvos em workspaces de fedavg e fedprox.
"""

import os
import glob
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from pathlib import Path

# Imports dos modelos e data loaders
from custom.models.experiment.CNN import CNN
from custom.models.experiment.MLP import MLP
from custom.utils.image_data_loader import load_val_data as load_val_data_image
from custom.utils.tabular_data_loader import load_val_data as load_val_data_tabular
from custom.utils.tabular_data_loader import DATASET_CONFIG as TABULAR_CONFIG


def validate(model, val_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            if data.dtype == torch.float16:
                data = data.float()
            output = model(data)
            loss = criterion(output, target)

            loss_sum += loss.item()
            num_batches += 1

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    val_accuracy = correct / total if total > 0 else 0.0
    val_loss = loss_sum / num_batches if num_batches > 0 else 0.0
    return val_accuracy, val_loss


def get_model_for_dataset(dataset_name, device):
    if dataset_name in ['adult', 'covtype']:
        config = TABULAR_CONFIG[dataset_name]
        model = MLP(
            input_size=config['input_shape'][0],
            num_classes=config['num_classes']
        )
    elif dataset_name in ['mnist', 'fmnist']:
        model = CNN(input_channels=1, input_size=28, num_classes=10)
    elif dataset_name in ['cifar10', 'svhn']:
        model = CNN(input_channels=3, input_size=32, num_classes=10)
    else:
        raise ValueError(f"Dataset {dataset_name} não suportado")
    
    model.to(device)
    return model


def get_val_loader_for_dataset(dataset_name, site_name):
    
    if dataset_name in ['adult', 'covtype']:
        val_loader = load_val_data_tabular(
            site_name=site_name,
            dataset_name=dataset_name,
            batch_size=64,
            num_clients=10,
            alpha=0.5,
            skew_type="quantity"
        )
    else:
        val_loader = load_val_data_image(
            site_name=site_name,
            dataset_name=dataset_name,
            batch_size=64
        )
    return val_loader


def extract_info_from_path(model_path):
    parts = Path(model_path).parts
    
    workspace_idx = None
    for i, part in enumerate(parts):
        if part.startswith('workspace_'):
            workspace_idx = i
            break
    
    if workspace_idx is None:
        raise ValueError(f"Não foi possível encontrar workspace no caminho: {model_path}")
    
    workspace_name = parts[workspace_idx] 
    dataset_name = workspace_name.replace('workspace_', '')  
    
    if 'fedavg' in parts:
        algorithm = 'fedavg'
    elif 'fedprox' in parts:
        algorithm = 'fedprox'
    else:
        algorithm = 'unknown'
    
   
    site_name = f"site-1"
    
    return {
        'dataset_name': dataset_name,
        'algorithm': algorithm,
        'workspace_name': workspace_name,
        'site_name': site_name,
        'model_path': model_path
    }


def evaluate_model(model_path, device):
    try:
        info = extract_info_from_path(model_path)
        dataset_name = info['dataset_name']
        algorithm = info['algorithm']
        site_name = info['site_name']
        
        print(f"\n{'='*70}")
        print(f"📊 Avaliando: {algorithm} - {dataset_name}")
        print(f"   Caminho: {model_path}")
        print(f"{'='*70}")
        
        state = torch.load(model_path, map_location=device)
        
        if 'model' in state:
            model_state = state['model']
        else:
            model_state = state
        
        model = get_model_for_dataset(dataset_name, device)
        model.load_state_dict(model_state)
        model.eval()
        
        val_loader = get_val_loader_for_dataset(dataset_name, site_name)
        
        criterion = nn.CrossEntropyLoss()
        val_accuracy, val_loss = validate(model, val_loader, device, criterion)
        
        print(f"✅ Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")
        
        return {
            'dataset': dataset_name,
            'algorithm': algorithm,
            'workspace': info['workspace_name'],
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'model_path': model_path,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"❌ Erro ao avaliar {model_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': info.get('dataset_name', 'unknown'),
            'algorithm': info.get('algorithm', 'unknown'),
            'workspace': info.get('workspace_name', 'unknown'),
            'val_accuracy': None,
            'val_loss': None,
            'model_path': model_path,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e)
        }


def main():
    base_dir = Path("/home/marcelohaps/NVFlare-ResIA/custom/2102.02079_experiment/quantity_skew")
    output_dir = Path("/home/marcelohaps/NVFlare-ResIA/custom/2102.02079_experiment/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Usando device: {device}")
    
    model_pattern = "**/best_FL_global_model.pt"
    model_paths = []
    
    for algorithm_dir in ['fedavg', 'fedprox']:
        algorithm_path = base_dir / algorithm_dir
        if algorithm_path.exists():
            paths = list(algorithm_path.glob(model_pattern))
            model_paths.extend(paths)
            print(f"📁 Encontrados {len(paths)} modelos em {algorithm_dir}")
    
    print(f"\n📊 Total de modelos encontrados: {len(model_paths)}")
    
    results = []
    for i, model_path in enumerate(sorted(model_paths), 1):
        print(f"\n[{i}/{len(model_paths)}] Processando...")
        result = evaluate_model(str(model_path), device)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    output_file = output_dir / "global_model_evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"✅ Resultados salvos em: {output_file}")
    print(f"{'='*70}")
    
    print("\n📊 RESUMO DOS RESULTADOS:")
    print(df.groupby(['dataset', 'algorithm'])[['val_accuracy', 'val_loss']].mean())
    
    output_json = output_dir / "global_model_evaluation_results.json"
    df.to_json(output_json, orient='records', indent=2)
    print(f"\n💾 Resultados também salvos em JSON: {output_json}")


if __name__ == "__main__":
    main()