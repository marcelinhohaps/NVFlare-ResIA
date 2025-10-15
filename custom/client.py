# client_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import nvflare.client as flare
import copy

from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from simple_model import SimpleModel  # Add this import
from data_loader import load_train_data, load_val_data  
from validation import validate  
from nvflare.client.tracking import SummaryWriter

def main():
    num_epochs = 5
    nubatch_size = 256
    learning_rate = 0.01
    mu = 0.01
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    flare.init()
    
    site_name = flare.get_site_name()
    print(f"🏥 Cliente: {site_name}")
    
    site_num = int(site_name.split('-')[1])  

    train_loader = load_train_data(site_name, nubatch_size, num_clients=site_num)
    val_loader = load_val_data(site_name, nubatch_size, num_clients=site_num)
    
    while flare.is_running():
        input_model = flare.receive()
        print(f"🔄 Rodada: {input_model.current_round}")
        
        model = SimpleModel()
        model.to(device)
        model.load_state_dict(input_model.params)
        
        model_global = copy.deepcopy(model)
        model_global.eval()
        for param in model_global.parameters():
            param.requires_grad = False
        
        criterion = nn.CrossEntropyLoss()
        criterion_fedprox = PTFedProxLoss(mu=mu)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        num_steps = 0
        model.train()
        
        print(f"🚀 Iniciando treinamento com {num_epochs} épocas...")
        print(f"📊 Amostras de treino: {len(train_loader.dataset)}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_fedprox_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(data)
                loss_main = criterion(outputs, target)
                loss_fedprox = criterion_fedprox(model, model_global)
                loss = loss_main + loss_fedprox
                
                loss.backward()
                optimizer.step()
                
                num_steps += 1
                epoch_loss += loss_main.item()
                epoch_fedprox_loss += loss_fedprox.item()
            
            avg_loss = epoch_loss / len(train_loader)
            avg_fedprox = epoch_fedprox_loss / len(train_loader)
            print(f"  Época [{epoch+1}/{num_epochs}] - "
                  f"Loss: {avg_loss:.4f}, FedProx: {avg_fedprox:.4f}")
        
        val_accuracy, val_loss = validate(model, val_loader, device)
        print(f"✅ Validação: Accuracy={val_accuracy:.4f}, Loss={val_loss:.4f}")
        
        output_params = model.cpu().state_dict()
        
        summary_writer = SummaryWriter()
        summary_writer.add_scalar(tag="val_accuracy", scalar=val_accuracy, global_step=num_steps)
        summary_writer.add_scalar(tag="val_loss", scalar=val_loss, global_step=num_steps)
        
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