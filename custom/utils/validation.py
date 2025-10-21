"""
Funções de validação para o modelo
"""
import torch
import torch.nn as nn


def validate(model, val_loader, device):
    """
    Valida o modelo no conjunto de validação.
    
    Args:
        model: Modelo PyTorch
        val_loader: DataLoader de validação
        device: Device (cpu ou cuda)
        
    Returns:
        tuple: (accuracy, average_loss)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    model.train()
    
    return accuracy, avg_loss
