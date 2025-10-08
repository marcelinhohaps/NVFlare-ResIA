<img src="docs/resources/nvidia_eye.wwPt122j.png" alt="NVIDIA Logo" width="200">

# NVIDIA FLARE

[Website](https://nvidia.github.io/NVFlare) | [Artigo](https://arxiv.org/abs/2210.13291) | [Blogs](https://developer.nvidia.com/blog/tag/federated-learning) | [Talks & Papers](https://nvflare.readthedocs.io/en/main/publications_and_talks.html) | [Pesquisa](./research/README.md) | [Documentação](https://nvflare.readthedocs.io/en/main)

[![Blossom-CI](https://github.com/NVIDIA/nvflare/workflows/Blossom-CI/badge.svg?branch=main)](https://github.com/NVIDIA/nvflare/actions)

[![license](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](./LICENSE)
[![pypi](https://badge.fury.io/py/nvflare.svg)](https://badge.fury.io/py/nvflare)
[![pyversion](https://img.shields.io/pypi/pyversions/nvflare.svg)](https://badge.fury.io/py/nvflare)
[![downloads](https://static.pepy.tech/badge/nvflare)](https://pepy.tech/project/nvflare)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NVFlare)

[NVIDIA FLARE](https://nvidia.github.io/NVFlare/) (**NV**IDIA **F**ederated **L**earning **A**pplication **R**untime **E**nvironment)
É um SDK Python de código aberto, extensível e independente de domínio que permite a pesquisadores e cientistas de dados adaptar fluxos de trabalho existentes de ML/DL para um paradigma federado.
Ele possibilita que desenvolvedores de plataforma criem uma solução segura e que preserve a privacidade para colaboração distribuída entre múltiplas partes.

## Funcionalidades
O Flare é construído sobre uma arquitetura componentizada que permite levar cargas de trabalho de aprendizado federado desde a pesquisa e simulação até a implantação em produção no mundo real!

Funcionalidades de Aplicação
* Suporte tanto a algoritmos de aprendizado profundo quanto de aprendizado de máquina tradicional (ex: Pytorch, TensorFlow, Scikit-learn, XGBoost etc.)
* Suporte a aprendizado federado horizontal e vertical
* Algoritmos de aprendizado federado integrados (ex: FedNova, FedAvg, FedProx, SCAFFOLD etc.)
* Suporte a múltiplos fluxos de trabalho de treinamento controlados por servidor e cliente (ex: scatter & gather, cíclico) e fluxos de validação (avaliação de modelo global, validação entre usuários)
* Suporte tanto para análise de dados (estatísticas federadas) quanto para gerenciamento do ciclo de vida de aprendizado de máquina
* Preservação de privacidade com privacidade diferencial, criptografia homomórfica e interseção privada de conjuntos (PSI)

Da Simulação ao Mundo Real
* API de Cliente do FLARE para fazer a transição de ML/DL para FL com mínimas alterações de código
* Simulador e modo POC para desenvolvimento e prototipagem rápida
* Componentes totalmente personalizáveis e extensíveis com design modular
* Implantação em nuvem e on-premise
* Painel de controle para gerenciamento de projetos e implantação
* Reforço de segurança através de autorização federada e política de privacidade
* Suporte integrado para resiliência do sistema e tolerância a falhas


## Instalação
Para instalar a [versão atual](https://pypi.org/project/nvflare/):
```
$ python -m pip install nvflare
```

Para mais detalhes sobre a instalação, acesse: [NVIDIA FLARE installation](https://nvflare.readthedocs.io/en/main/installation.html).

## Simulador de Ambiente Federado

O objetivo do Simulador de Ambiente Federado é permitir que pesquisadores acelerem o processo de desenvolvimento de fluxos de trabalho de Aprendizado Federado (FL).

O Simulador de FL é um simulador leve de uma implantação de FL do NVFLARE em execução. Ele permite que pesquisadores testem e depurem suas aplicações sem precisar provisionar um projeto real. Os trabalhos de FL são executados em um servidor e em múltiplos clientes dentro do mesmo processo, mas de forma semelhante a como rodariam em uma implantação real. Isso possibilita que os pesquisadores desenvolvam novos componentes e trabalhos de forma mais rápida, podendo utilizá-los diretamente em uma implantação real em produção.

<!-- O FL Simulator com Recipe é a forma moderna e simplificada de executar simulações no NVIDIA FLARE usando a classe ```SimEnv``` combinada com Recipes como ```FedAvgRecipe``` -->

## Método CLI Tradicional (```nvflare-simulator```)

### Comandos

```
usage: nvflare simulator [-h] -w WORKSPACE [-n N_CLIENTS] [-c CLIENTS] [-t THREADS] [-gpu GPU] job_folder

argumentos posicionais:
job_folder -> pasta contendo o job a ser executado

argumentos opcionais:
-h, --help             mostra a mensagem de ajuda com alguns comandos
-w WORKSPACE, --workspace WORKSPACE
                       pasta do WORKSPACE (local de trabalho)
-n N_CLIENTS, --n_clients N_CLIENTS
                       número de clientes (usuários ou nós)
-c CLIENTS, --clients CLIENTS
                       lista do nome dos clientes
-t THREADS, --threads THREADS
                       número de clientes que podem ser rodados em paralelo
-gpu GPU, --gpu GPU
                       lista de IDs de GPU, separado por vírgula
-m MAX_CLIENTS, --max_clients MAX_CLIENTS
                       número máximo de clientes (padrão: 100)
```

### Estrutura do diretório ```job_folder```

```
├── my_job  
│   ├── app  
│   │   ├── config  
│   │   │   ├── config_client.json  
│   │   │   └── config_server.json  
│   │   └── custom  
│   └── meta.json 
```
Na pasta [job_templates](https://github.com/NVIDIA/NVFlare/tree/main/job_templates) há exemplos práticos de estruturas de jobs

### Capacidades e Limites

#### Limites de clientes

* Padrão: 100 clientes
* Poder ser aumentado com -m para mais de 100
* Limitado apenas pelos recursos da máquina

#### Modos de execução

* T = N: Cada cliente em um processo separado simultaneamente
* T < N: Clientes fazem swap-in/swap-out conforme processos ficam disponíveis

### Suporte Multi-GPU

O simulador distribui clientes entre GPUs de forma round-robin. Por exemplo, com ```-gpu 0,1``` e 5 clientes, os clientes são distribuídos alternadamente entre GPU 0 e GPU 1.

### Configurações de Clientes

Três formas de especificar clientes:

1. Lista explícita (```-c client1,client2```): Usa nomes específicos
2. Número (```-n 8```): Cria automaticamente site-1, site-2, etc.
3. Automático: Extrai do ```deploy_map``` do meta.json

### Limitações

O simulador não suporta:

* Provisionamento com certificados e segurança TLS
* Arquivo ```resources.json``` (use variáveis de ambiente)
* Autenticação e autorização completas
* Comunicação segura entre componentes

### Fluxo de Execução

Quando você roda o simulador:

1. Cria o servidor simulador
2. Inicia o admin server local
3. Cria e registra todos os clientes
4. Deploy e inicia a aplicação do servidor
5. Executa clientes sequencialmente ou em paralelo
6. Cada cliente busca tarefas, executa, e submete resultados de volta para o servidor


### Exemplos

#### Rodando uma Simulação

Este comando executará o mesmo aplicativo ```hello-numpy-sag``` no servidor e em 60 clientes, usando 60 threads.
Os nomes dos clientes serão site-1, site-2, … , site-60:

```nvflare simulator NVFlare/examples/hello-numpy-sag/app -w /tmp/nvflare/workspace_folder/ -n 60 -t 60```


## Método JobRecipe (ex: ```SimEnv + FedAvgRecipe```)

O JobRecipe usa uma API Python que abstrai a criação do job e execução do simulador. É a maneira mais fácil de utilizar o FL Simulator. Foi lançada na versão v2.7.0 do NVIDIA-FLARE e ainda está em technical review. Abaixo, está um exemplo do script de definição do job que configura tanto servidor quanto clientes usando o Recipe ```job.py```:

```
import argparse

from model import SimpleNetwork

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds

    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=SimpleNetwork(),
        train_script="client.py",
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()


if __name__ == "__main__":
    main()

```

### Parâmetros Obrigatórios de ```FedAvgRecipe```:

* ```name (str)```: Nome único do job de FL
* ```initial_model (any)```: Modelo Pytorch inicial para começar o treinamento federado. Pode ser ```None``` (clientes começam com modelos locais). Geralmente é uma instância de ```nn.Module```
* ```min_clients (int)```: Número mínimo de clientes necessários para iniciar uma rodada. Se N_clients < min_clients, a rodada não inicia
* ```num_rounds (int)```: Número de rodadas de treinamento federado
* ```train_script (str)```: Caminho para o script de treinamento do cliente

### Parâmetros Opcionais de ```FedAvgRecipe```:

* ```train_args (str)```: Argumentos de linha de comando para o script (padrão: "")
* ```aggregator (Optional[Aggregator])```: Agregador customizado (padrão: ```InTimeAccumulateWeightedAggregator```)
* ```params_transfer_type (TransferType)```: Como transferir parâmetros - ```FULL``` (modelo completo) ou ```DIFF``` (apenas diferenças)
* ```launch_external_process (bool)```: Se deve lançar o script em processo externo (padrão: False)

### Parâmetros do ```SimEnv```:

* ```num_clients (int)```: Número de clientes simulados
* ```num_threads (Optional[int])```: Número de threads paralelos (padrão: igual a ```num_clients```)
* ```gpu_config (str)```: Configuração de GPUs (ex: "0,1")
* ```workspace_root (str)```: Diretório do workspace (padrão: ```/tmp/nvflare/simulation```)

### Cient API - Métodos Essenciais:

* ```flare.init()```: Inicializa o ambiente da Client API
* ```flare.receive```: Recebe o modelo global do servidor (retorna ```FLModel```)
* ```flare.send()```: Envia o modelo atualizado de volta ao servidor
* ```flare.is_running()```: Verifica se o job FL ainda está ativo

### Objeto ```Run```

O método ```recipe.execute(env)``` retorna um objeto ```Run``` que permite monitorar o job

* ```run.get_job_id()```: Retorna o ID do Job
* ```run.get_status()```: Retorna o status da execução
* ```run.get_result()```: Retorna o caminho dos resultados



Para o lado do cliente, o código ```client.py``` seria:

```

import os
import torch
from model import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

DATASET_PATH = "/tmp/nvflare/data"


def main():
    batch_size = 16
    epochs = 2
    lr = 0.01
    model = SimpleNetwork()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    train_dataset = CIFAR10(
        root=os.path.join(DATASET_PATH, client_name), transform=transforms, download=True, train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    summary_writer = SummaryWriter()
    while flare.is_running():
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")

        model.load_state_dict(input_model.params)
        model.to(device)

        steps = epochs * len(train_loader)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()

                predictions = model(images)
                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                if i % 3000 == 0:
                    print(f"site={client_name}, Epoch: {epoch}/{epochs}, Iteration: {i}, Loss: {running_loss / 3000}")
                    global_step = input_model.current_round * steps + epoch * len(train_loader) + i
                    summary_writer.add_scalar(
                        tag="loss_for_each_batch", scalar=float(running_loss), global_step=global_step
                    )
                    running_loss = 0.0

        print(f"Finished Training for {client_name}")

        PATH = "./cifar_net.pth"
        torch.save(model.state_dict(), PATH)

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        flare.send(output_model)


if __name__ == "__main__":
    main()
```

Para executar o simulador, podemos apenas executar o código python referente ao job ```job.py```:

```
python3 job.py
```

Nota: Essa é a maneira mais fácil e menos abstrata de executar uma simulação no NVIDIA-FLARE. Perceba pelos códigos que não temos tanto controle das configurações.

### Notas sobre Recipes Disponíveis

Nem todos os algoritmos de FL têm recipes ainda. Essas são as disponíveis:

* ```FedAvgRecipe``` (PyTorch)
* ```FedAvgRecipe``` (TensorFlow)
* ```NumpyFedAvgRecipe```
* ```FedAvgLrRecipe``` (Regressão Logística)
* ```FlowerRecipe``` (integração com Flower)

Nota: Algoritmos como FedProx, FedNova e SCAFFOLD ainda não têm recipes dedicados e devem ser utilizados via método CLI tradicional.


## License

NVIDIA FLARE is released under an [Apache 2.0 license](./LICENSE).
