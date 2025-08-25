import argparse
import torch
import yaml
from esercizio1 import esercizio1
from esercizio2 import esercizio2

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run NLP experiments")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["1", "2"],
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
        "2": esercizio2
    }

    experiment_fn = experiment_map[args.experiment]

    #Carica la configurazione yaml se è l'esercizio 2
    if experiment_fn == esercizio2:
        config_path = f"./config/config_exp{args.experiment}.yaml"
        config = load_config(config_path)
        config["device"] = device
        experiment_fn(config)
    else:
        experiment_fn()

if __name__ == "__main__":
    main()
