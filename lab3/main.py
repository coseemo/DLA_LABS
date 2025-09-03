import argparse
import torch
import yaml
from esercizio1 import esercizio1
from esercizi2e3 import esercizi2e3

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Transformers experiments")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["1", "2e3"],
                        help="Which experiment to run")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use for training")
    args = parser.parse_args()

    #Setto il device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    if device == "cuda":
        torch.cuda.empty_cache()

    #Mappa gli esperimenti
    experiment_map = {
        "1": esercizio1,
        "2e3": esercizi2e3
    }

    experiment_fn = experiment_map[args.experiment]

    #Carica la configurazione yaml se non è il primo esercizio
    if experiment_fn == esercizi2e3 :
        config_path = f"./config/config_exp{args.experiment}.yaml"
        config = load_config(config_path)
        config["device"] = device
        experiment_fn(config)
    else:
        experiment_fn()

if __name__ == "__main__":
    main()
