# client_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import nvflare.client as flare
import copy

from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from custom.models.experiment.CNN import CNN
from custom.models.experiment.MLP import MLP
from custom.utils.tabular_data_loader import load_train_data, load_val_data  
from nvflare.client.tracking import SummaryWriter, WandBWriter


def validate(model, val_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
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

def main():
    num_epochs = 10
    nubatch_size = 64
    learning_rate = 0.01
    mu = 0.01   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = 'covtype'
    flare.init()
    site_name = flare.get_site_name()
    
    train_loader = load_train_data(
        site_name, 
        dataset_name=dataset_name, 
        batch_size=nubatch_size, 
        num_clients=10, 
        alpha=0.5, 
        skew_type="quantity"
    )
    val_loader = load_val_data(
        site_name, 
        dataset_name=dataset_name, 
        batch_size=nubatch_size,
        num_clients=10,    
        alpha=0.5,         
        skew_type="quantity" 
    )

    wandb_writer = WandBWriter()
    
    while flare.is_running():
        input_model = flare.receive()
        rodada = input_model.current_round
        
        if dataset_name == 'covtype':
            model = MLP(54, 7)
        elif dataset_name == 'adult':
            model = MLP(100, 2)
        elif dataset_name == 'mnist':
            model = CNN(1, 28, 10)
        elif dataset_name == 'cifar10':
            model = CNN(3, 32, 10)
        elif dataset_name == 'svhn':
            model = CNN(3, 32, 10)
        elif dataset_name == 'fmnist':
            model = CNN(1, 28, 10)
        model.to(device)
        model.load_state_dict(input_model.params)
        
        model_global = copy.deepcopy(model)
        model_global.eval()
        for param in model_global.parameters():
            param.requires_grad = False
        
        criterion = nn.CrossEntropyLoss()
        criterion_prox = PTFedProxLoss(mu=mu)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        num_steps = 0
        model.train()
        
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            fed_prox_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(data)
                loss_main = criterion(outputs, target)
                loss = loss_main
                fed_prox_loss = criterion_prox(model, model_global)
                loss += fed_prox_loss
                loss.backward()
                optimizer.step()
                
                num_steps += 1
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"  Época [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
        
        val_accuracy, val_loss = validate(model, val_loader, device, criterion)
        wandb_writer.log(metrics={"val_accuracy": val_accuracy, "val_loss": val_loss}, step=rodada)
        
        output_params = model.cpu().state_dict()
        
        
        
        flare.send(
            flare.FLModel(
                params=output_params,
                meta={
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "NUM_STEPS_CURRENT_ROUND": num_steps,
                    "initial_metrics": {
                        "accuracy": val_accuracy,
                        "val_loss": val_loss
                    },
                    "validation_metric": val_accuracy
                }
            )
        )
        
        print(f"📤 Modelo enviado: {num_steps} steps, accuracy={val_accuracy:.4f}")


if __name__ == "__main__":
    main() 