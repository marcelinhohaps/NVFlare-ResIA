import argparse

from custom.models.experiment.CNN import CNN
from custom.models.experiment.MLP import MLP
#from fed_hybrid_recipe import FedHybridRecipe
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser(
        description="FedAVG"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="simulator",
        choices=["simulator", "export", "poc"],
        help="Modo de execução: simulator (local), export (gera job), ou poc"
    )
    parser.add_argument(
        "--n_clients", 
        type=int, 
        default=10,
        help="Número de clientes"
    )
    parser.add_argument(
        "--num_rounds", 
        type=int, 
        default=50,
        help="Número de rodadas de FL"
    )
    
    parser.add_argument(
        "--min_accuracy_weight",
        type=float,
        default=0.1,
        help="Peso mínimo para clientes com baixa acurácia"
    )
    parser.add_argument(
        "--no_normalize_alpha",
        action="store_true",
        help="Desabilita normalização de alphas (padrão: normaliza)"
    )
    
    parser.add_argument(
        "--job_name",
        type=str,
        default="fed_hybrid_job",
        help="Nome do job"
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="/tmp/nvflare/jobs/fed_hybrid",
        help="Caminho para exportar job (modo export)"
    )
    
    parser.add_argument(
        "--tracking",
        type=str,
        default="wandb",
        choices=["tensorboard", "mlflow", "wandb", "none"],
        help="Tipo de experiment tracking"
    )
    
    return parser.parse_args()


def main():
    args = define_parser()
    dataset_name = 'svhn'
    print("=" * 80)
    print("🚀 FedProx")
    print("=" * 80)
    print(f"Modo: {args.mode}")
    print(f"Clientes: {args.n_clients}")
    print(f"Rodadas: {args.num_rounds}")
    print(f"Min accuracy weight: {args.min_accuracy_weight}")
    print(f"Normalize alpha: {not args.no_normalize_alpha}")
    print("=" * 80)
    
    initial_model = CNN()
    print(f"✅ Modelo inicial criado: {initial_model.__class__.__name__}")
    
    recipe = FedAvgRecipe(
        name=f"{dataset_name}_fedprox_quantity-skew",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        initial_model=initial_model,
        train_script="/home/marcelohaps/NVFlare-ResIA/custom/2102.02079_experiment/quantity_skew/fedprox/client.py",
    )
    add_experiment_tracking(
    recipe,
    tracking_type="wandb",
    tracking_config={
        "mode": "online", 
        "wandb_args": {
            "project": f"{dataset_name.upper()}_FEDPROX_QUANTITY-SKEW",
            "group": f"{dataset_name.upper()}_FEDPROX_QUANTITY-SKEW",
            "job_type": "train",
            "name": f"{dataset_name}-fedprox",  
            "notes": "experimento base",
            "tags": [dataset_name, "fedprox", "quantity-skew"],
            "config": {"lr": 0.01, "batch_size": 64, "alpha": 0.5, "n_clients": args.n_clients},
        },
    },
)
    
    if args.mode == "simulator":
        print("\n" + "=" * 80)
        print("🏃 Executando em modo SIMULADOR (local)")
        print("=" * 80 + "\n")
        
        env = SimEnv(
            num_clients=args.n_clients, 
            num_threads=args.n_clients,
            workspace_root=f"/home/marcelohaps/NVFlare-ResIA/custom/2102.02079_experiment/quantity_skew/fedprox/workspace_{dataset_name}"
        )
        
        run = recipe.execute(env)
        
        print("\n" + "=" * 80)
        print("📊 RESULTADOS")
        print("=" * 80)
        print(f"Status: {run.get_status()}")
        print(f"Resultado: {run.get_result()}")
        print("=" * 80 + "\n")
        
    elif args.mode == "export":
        print("\n" + "=" * 80)
        print("📦 Exportando job")
        print("=" * 80 + "\n")
        
        recipe.export_job(args.export_path)
        
        print(f"✅ Job exportado para: {args.export_path}")
        print(f"\nPara executar:")
        print(f"  nvflare job submit {args.export_path}")
        print()
        
    elif args.mode == "poc":
        print("\n" + "=" * 80)
        print("🧪 Modo POC (Proof of Concept)")
        print("=" * 80 + "\n")
        
        poc_path = f"/tmp/nvflare/poc/{dataset_name}_fedprox_quantity-skew"
        recipe.export_job(poc_path)
        
        print(f"✅ Job exportado para POC: {poc_path}")
        print(f"\nPara executar:")
        print(f"  1. nvflare poc prepare -n {args.n_clients}")
        print(f"  2. nvflare job submit {poc_path}")
        print()


if __name__ == "__main__":
    main()

