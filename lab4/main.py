import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb
import matplotlib.pyplot as plt
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from dataloaders import CIFAR10_DataLoader, FakeDataLoader
from logger import Logger
from torch.utils.data import DataLoader, Dataset, random_split
from models import CNN, CNN_plus, Autoencoder
from metrics import plot_confusion_matrix_accuracy, plot_score, plot_logit_softmax, compute_scores, max_logit, max_softmax
from fgsm import FGSM_trainer


#Imposta il seed per riproducibilità
def set_seed(seed):
    #Seed per Python random
    random.seed(seed)
    #Seed per NumPy
    np.random.seed(seed)
    #Seed per PyTorch CPU
    torch.manual_seed(seed)
    #Seed per PyTorch GPU (se disponibile)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

#Carica le configurazioni da un file yaml
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

#Assembla il modello utilizzando le specifiche del file yaml
def create_model(model_name):
    if model_name == "CNN":
        model = CNN()
    elif model_name == "CNNplus":
        model = CNN_plus()
    elif model_name == "Autoencoder":
        model = Autoencoder()
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    return model

#Funzione per il pretrain per il singolo modello
def pretrain_single_model(model_config, global_config, data_loaders):
    
    device = global_config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device
    torch.cuda.empty_cache() if device == "cuda" else None

    train_dl, val_dl, test_dl = data_loaders

    #Estraggo il tipo di modello dal nome
    model_type = model_config["name"].split("_")[0]
    model = create_model(model_type)
    model.to(device)

    #Setup del logger
    logger = Logger(
        project_name=global_config["project_name"],
        run_name=f"pretrain_{model_config['name']}",
        config={**global_config, **model_config}
    )

    #Inizializzo il trainer
    trainer = FGSM_trainer(
        model=model,
        mean=global_config["mean"],
        std=global_config["std"], 
        logger=logger,
        device=device
    )
    
    #Faccio il setup del training
    trainer.setup_training(
        optimizer=model_config["optimizer"],
        lr=model_config["lr"],
        scheduler=model_config["scheduler"],
        max_iter=model_config["epochs"],
        fgsm=model_config["fgsm"],
        epsilon=model_config["epsilon"]
    )

    #Addestramento
    print("Starting training...")
    for epoch in range(model_config["epochs"]):
        if model.type == "classifier":
            train_loss_clean, train_loss_adv = trainer.train_classifier(train_dl, epoch)
            val_acc, val_loss = trainer.test_classifier(val_dl)
        else:  # autoencoder
            train_loss_clean, train_loss_adv = trainer.train_autoencoder(train_dl, epoch)
            val_scores, val_loss = trainer.test_autoencoder(val_dl)
        
        # Aggiorna scheduler
        if trainer.scheduler:
            trainer.scheduler.step()
        
        #Log delle metriche
        if logger:
            metrics = {
                "train/loss_clean": train_loss_clean,
                "train/loss_adv": train_loss_adv,
                "val/loss": val_loss,
                "epoch": epoch
            }
            logger.log_metrics(metrics)
        
        print(f"Epoch {epoch+1}/{model_config['epochs']}: "
              f"Train Loss Clean: {train_loss_clean:.4f}, "
              f"Train Loss Adv: {train_loss_adv:.4f}, "
              f"Val Loss: {val_loss:.4f}")

    #Test finale
    if model.type == "classifier":
        test_acc, test_loss = trainer.test_classifier(test_dl)
        logger.log_metrics({"test/accuracy": test_acc})
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    else:
        test_scores, test_loss = trainer.test_autoencoder(test_dl)
        logger.log_metrics({"test/scores": test_scores})
        print(f"Test Loss: {test_loss:.4f}")

    #Salvo il modello
    save_path = model_config["path"]
    base_dir = os.path.dirname(save_path)
    os.makedirs(base_dir, exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

    logger.finish()

#Funzione principale per il pretrain di tutti i modelli configurati in config_pretrain.yaml
def pretrain(config):
    
    print("Starting Pretraining..")

    global_configs = config["global_configs"]
    set_seed(global_configs["seed"])
    
    #Carico i dati 
    data_loader = CIFAR10_DataLoader(
        batch_size=global_configs["batch_size"],
        split=global_configs["validation_split"],
        num_workers=global_configs["num_workers"]
    )
    train_dl, val_dl, test_dl = data_loader.get_dataloaders()
    data_loaders = (train_dl, val_dl, test_dl)
    
    #Pretrain dei modelli CNN 
    if config["pretraining_configs"]["pretrain_cnn"]:
        print("Pretraining CNN models...")
        cnn_models = config["pretraining_configs"]["cnn_models"]
        for model_config in cnn_models:
            pretrain_single_model(model_config, global_configs, data_loaders)
    
    #Pretrain dei modelli CNNplus
    if config["pretraining_configs"]["pretrain_cnn_plus"]:
        print("Pretraining CNNplus models...")
        cnn_plus_models = config["pretraining_configs"]["cnn_plus_models"]
        for model_config in cnn_plus_models:
            pretrain_single_model(model_config, global_configs, data_loaders)
    
    #Pretrain dei modelli Autoencoder 
    if config["pretraining_configs"]["pretrain_autoencoder"]:
        print("Pretraining Autoencoder models...")
        autoencoder_models = config["pretraining_configs"]["autoencoder_models"]
        for model_config in autoencoder_models:
            pretrain_single_model(model_config, global_configs, data_loaders)
    
    print("Pretraining Finished.")

#Metodo dedicato per gli esercizi 1 e 2
def experiment_1e2(config):
    print("Running Experiment 1 and 2: OOD Detection")
    
    #Device e seed
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device
    torch.cuda.empty_cache() if device == "cuda" else None
    
    set_seed(config["seed"])
    
    #Carico i dati clean
    data_loader = CIFAR10_DataLoader(
        batch_size=config["data"]["batch_size"],
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    train_dl, val_dl, test_dl = data_loader.get_dataloaders()
    
    #Carico i dati fake
    fake_data_loader = FakeDataLoader(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"]
    )
    fake_dl = fake_data_loader.get_dataloader()

    test_dl_fgsm = DataLoader(test_dl.dataset, batch_size=1, shuffle=False)
    
    #Creo la directory dei plot
    os.makedirs("plots/es1e2/", exist_ok=True)
    
    #Testo per le CNN
    if config["models"]["test_cnn"]:
        print("Testing CNN models...")

        #I modelli testati sono quelli nel file config_1e2.yaml
        cnn_models = config["models"]["cnn_models"]
        
        for model_config in cnn_models:
            model_name = model_config["name"]
            model_path = model_config["path"]

            save_path = f"plots/es1e2/{model_name}"
            os.makedirs( save_path, exist_ok=True)
            
            print(f"Testing {model_name}...")
            
            #Creazione del modello a partire dal nome
            model_type = model_name.split("_")[0]  
            model = create_model(model_type)
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            #Setup del trainer
            trainer = FGSM_trainer(
                model=model,
                mean=config["data"]["mean"],
                std=config["data"]["std"],
                device=device
            )
            
            #Prendo le predizioni per la matrice di confusione
            y_gt, y_pred = trainer.get_pred_classifier(test_dl)
            
            #Plot della matrice di confusione e dell'accuracy
            plot_confusion_matrix_accuracy(
                y_gt, y_pred, test_dl, model_name, 
                config["evaluation"]["temperature"],
                save_path = save_path
            )
            
            #Prendo dei batch di campione per i plot di logit/softmax
            test_batch = next(iter(test_dl))
            fake_batch = next(iter(fake_dl))
            x_test, y_test = test_batch[0], test_batch[1]
            x_fake, _ = fake_batch[0], fake_batch[1] if len(fake_batch) > 1 else fake_batch[0]

            #Indice del campione
            k = 0 
            
            #Plot logit/softmax per i dati puliti
            plot_logit_softmax(
                x_test, k, model, device, model_name, 
                config["evaluation"]["temperature"],
                save_path = save_path, ty="Real_data"
            )
            
            #Plot logit/softmax per i dati fake
            plot_logit_softmax(
                x_fake, k, model, device, model_name,
                config["evaluation"]["temperature"], 
                save_path = save_path, ty="Fake_data"
            )
            
            #Calcolo gli scores con max_logit
            print("Computing scores with max_logit...")
            scores_test = compute_scores(model, test_dl, max_logit, device)
            scores_fake = compute_scores(model, fake_dl, max_logit, device)
            plot_score(
                scores_test, scores_fake, model_name,
                config["evaluation"]["temperature"],
                save_path = save_path, score_fun="max_logit"
            )
            
            #Calcolo gli scores usando max_softmax con temperatura
            print("Computing scores with max_softmax...")
            temp = config["evaluation"]["temperature"]
            scores_test = compute_scores(
                model, test_dl, 
                lambda l: max_softmax(l, t=temp), 
                device
            )
            scores_fake = compute_scores(
                model, fake_dl, 
                lambda l: max_softmax(l, t=temp), 
                device
            )
            plot_score(
                scores_test, scores_fake, model_name,
                config["evaluation"]["temperature"],
                save_path = save_path, score_fun="max_softmax"
            )

            print(f"Running FGSM attack tests for {model_name}...")
            
            epsilons = config["fgsm"]["epsilons_cnn"]
            accuracies = []
            examples = []
            
            for eps in epsilons:
                trainer.setup_attack(eps)
                acc, ex = trainer.test_attack_classifier(test_dl_fgsm)
                accuracies.append(acc)
                examples.append(ex)
                print(f"Epsilon: {eps:.3f} -> Accuracy: {acc:.4f}")
            
            #Plot accuracy vs epsilon
            trainer.plot_result(
                epsilons=epsilons,
                examples=examples,
                metric=accuracies,
                model_name="CNN",
                save_path=save_path
            )

            print(f"Completed testing {model_name}")
    
    #Test per gli autoencoder
    if config["models"]["test_autoencoder"]:
        print("Testing Autoencoder...")
        
        ae_models = config["models"]["autoencoder"]

        for model_config in ae_models:
            
            model_name = model_config["name"]
            model_path = model_config["path"]

            save_path = f"plots/es1e2/{model_name}"
            os.makedirs( save_path, exist_ok=True)
            
            #Creao il modello
            model = create_model("Autoencoder")
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            #Setup del trainer
            trainer = FGSM_trainer(
                model=model,
                mean=config["data"]["mean"],
                std=config["data"]["std"],
                device=device
            )
            
            #Test dell'autoencoder sui dati di test e quelli fake
            test_scores, test_loss = trainer.test_autoencoder(test_dl)
            fake_scores, fake_loss = trainer.test_autoencoder(fake_dl)
            
            #Plotto il confronto tra gli scores
            plot_score(
                test_scores, fake_scores, model_name,
                config["evaluation"]["temperature"],
                save_path = save_path, score_fun="reconstruction_error"
            )

            print(f"Running FGSM attack tests for {model_name}...")
            
            epsilons = config["fgsm"]["epsilons_ae"]
            accuracies = []
            examples = []
            
            for eps in epsilons:
                trainer.setup_attack(eps)
                acc, ex = trainer.test_attack_autoencoder(test_dl_fgsm)
                accuracies.append(acc)
                examples.append(ex)
                print(f"Epsilon: {eps:.3f} -> MSE: {acc:.4f}")
            
            #Plot accuracy vs epsilon
            trainer.plot_result(
                epsilons=epsilons,
                examples=examples,
                metric=accuracies,
                model_name="Autoencoder",
                save_path=save_path
            )
            
            print(f"Autoencoder test loss: {test_loss:.4f}")
            print(f"Autoencoder fake loss: {fake_loss:.4f}")
            print("Completed testing Autoencoder")
    
    print("Experiment 1 and 2 completed. All plots saved in plots/es1e2/")
    

#Metodo dedicato per l'esercizio 3: Targeted FGSM Attack
def experiment_3(config):
    print("Running Experiment 3: Targeted FGSM Attack")

    #Device
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device
    torch.cuda.empty_cache() if device == "cuda" else None

    #Carico i dati per test
    data_loader = CIFAR10_DataLoader(
        batch_size=1,  # batch_size=1 per attacco mirato
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    _, _, test_dl = data_loader.get_dataloaders()

    # Parametri dai dati
    mean = config["data"]["mean"]
    std = config["data"]["std"]
        
    # Get CNN model names and paths from config
    cnn_models = config["models"]["cnn_models"]
    
    for model_config in cnn_models:
        model_name = model_config["name"]
        model_path = model_config["path"]
    
        #Creo il classificatore
        model_type = model_name.split("_")[0]  
        classifier = create_model(model_type)
        classifier.to(device)
        
        #Carico pesi pre-addestrati
        classifier.load_state_dict(torch.load(model_path, map_location=device))
        classifier.eval()
        
        #Logger
        logger = Logger(
            project_name=config["logging"]["project_name"],
            run_name=f"exp3_targeted_FGSM_{model_name}",
            config=config
        )
    
        #Creo trainer FGSM
        trainer = FGSM_trainer(
            model=classifier, 
            mean=mean, 
            std=std, 
            logger=logger,
            device=device
        )
    
        #Epsilons da testare
        epsilons = config["fgsm"]["epsilons_cnn"]
        target_class = config["fgsm"]["target_class"]  #Classe target per l'attacco (può essere modificata in config_3.yaml)
    
        all_results = []  #Lista completa dei risultati (dizionari)
        targeted_successes = []  #Solo i success rate per il plot
        all_examples = []
        
        tqdm_bar = tqdm(epsilons, total=len(epsilons), desc="[Targeted FGSM Testing]")
        for eps in tqdm_bar:
            trainer.setup_attack(eps)
            results, examples = trainer.targeted_attack_classifier(test_dl, target_class)
            
            all_results.append(results)
            targeted_successes.append(results['overall_success_rate'])  
            all_examples.append(examples)
            
            # Aggiorna la progress bar
            tqdm_bar.set_postfix(
                epsilon=f"{eps:.3f}", 
                success=f"{results['overall_success_rate']:.2%}"
            )

            trainer.print_attack_summary(results, target_class, eps)
        
        # Plot risultati
        save_path = f"plots/es3/{model_name}/"
        os.makedirs(save_path, exist_ok=True)

        #Plot successi vs eps
        trainer.plot_target_attack(
            epsilons, targeted_successes, target_class, model_name,
            save_path=save_path
        )
        
        # Plot esempi qualitativi
        for i, eps in enumerate([0.0, 0.05, 0.075, 0.1, 0.125, 0.15]):
            if eps in epsilons:
                idx = epsilons.index(eps)
                trainer.plot_targeted_examples(
                    all_examples[idx], eps, model_name, 
                    save_path=save_path
                )
        
        print(f"Targeted FGSM experiment completed. Plots saved in: {save_path}")
        logger.finish()



def main():
    parser = argparse.ArgumentParser(description="Run OOD experiments")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["1e2", "3", "pretrain"],
                       help="Which experiment to run")
    
    args = parser.parse_args()

    #Carico la configurazione
    config_path = f"./configs/config_{args.experiment}.yaml"
    config = load_config(config_path)
    
    #Si sceglie l'esperimento
    experiment_map = {
        "1e2": experiment_1e2,
        "3": experiment_3,
        "pretrain": pretrain
    }
    
    experiment_fn = experiment_map[args.experiment]
    experiment_fn(config)

if __name__ == "__main__":
    main()