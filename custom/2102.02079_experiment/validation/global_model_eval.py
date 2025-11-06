import torch
import torch.nn as nn
import torch.optim as optim
import nvflare.client as flare
import copy

from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from custom.models import CNN, MLP 
from custom.utils.data_loader import load_train_data, load_val_data  
from nvflare.client.tracking import SummaryWriter, WandBWriter
from custom.client import validate
import pandas as pd
from datetime import datetime
import json
import os


def global_model_eval_job(site_name, model, device):
    val_loader = load_val_data(site_name, dataset_name='mnist', batch_size=64)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_accuracy, val_loss = validate(model, val_loader, device, criterion)
    return val_accuracy, val_loss


def main():
    site_name = 'mnist_fedavg_quantity-skew'
    output = '/home/marcelohaps/NVFlare-ResIA/custom/validation'
    weight_path = '/home/marcelohaps/NVFlare-ResIA/custom/workspace_mnist/mnist_fedavg_quantity-skew/server/simulate_job/app_server/best_FL_global_model.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(weight_path, map_location=device)
    model = CNN()
    model.load_state_dict(state['model'])   
    model.eval()
    val_accuracy, val_loss = global_model_eval_job(site_name, model, device)
    
    print(f"Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")
    df = pd.DataFrame({'experiment_name': [site_name], 'val_accuracy': [val_accuracy], 'val_loss': [val_loss], 'date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]})
    with open(os.path.join(output, 'global_model_eval.json'), 'a') as f:
        df.to_json(f, orient='records', lines=True)
        f.write('\n')

if __name__ == "__main__":
    main()