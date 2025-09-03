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
from model_runner import Model_Runner
from torch.utils.data import DataLoader, Dataset, random_split
from models import CNN, CNN_discriminator, CNN_plus, Autoencoder
from metrics import plot_confusion_matrix_accuracy, plot_score, plot_logit_softmax, compute_scores, max_logit, max_softmax
from fgsm_and_jarn import FGSM_trainer


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
    elif model_name == "CNNdiscriminator":
        model = CNN_discriminator()
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    return model

#Funzione per il pretrain dei modelli se necessario
def pretrain(config):
    print("Running Pretraining")

    #Device
    device = config["device"]

    #Carico i dati
    data_loader = CIFAR10_DataLoader(
        batch_size=config["data"]["batch_size"],
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    train_dl, val_dl, test_dl = data_loader.get_dataloaders()

    model_name = os.path.splitext(os.path.basename(config["model"]["path"]))[0]
    if "_" in model_name:
        model_name = model_name.split("_")[-1]

    #Creo il modello a partire da config
    model = create_model(model_name)
    model.to(device)

    #Setup del logger
    logger = Logger(
        project_name=config["logging"]["project_name"],
        run_name=f"pretrain_{model_name}",
        config=config
    )

    #Inizializzo il runner
    runner = Model_Runner(model, logger)

    #Faccio il setup(loss, optimizer, scheduler ecc.)
    runner.setup(
        criterion=config["train"]["criterion"], 
        criterion_params=config["train"].get("criterion_params", None),
        optimizer=config["train"]["optimizer"],
        lr=config["train"]["lr"],
        scheduler=config["train"]["scheduler"],
        max_iter=config["train"]["epochs"]
    )

    #Addestramento
    train_losses, train_acc, val_losses, val_acc = runner.train(
        dl_train=train_dl,
        dl_val=val_dl,
        train_epochs_num=config["train"]["epochs"]
    )

    #Salvo il modello
    base_dir = os.path.dirname(config["model"]["path"])
    os.makedirs(base_dir, exist_ok=True)
    
    save_path = os.path.join(base_dir, model.type + ".pth")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

    
    #Test finale
    test_loss, test_acc, y_true, y_pred = runner.test(test_dl)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

    logger.finish()


#Metodo dedicato per l'esercizio 1: OOD Detection and Performance Evaluation
def experiment_1(config):
    print("Running Experiment 1: OOD Detection and Performance Evaluation")

    #Device
    device = config["device"]
    
    #Carico i dati
    data_loader = CIFAR10_DataLoader(
        batch_size=config["data"]["batch_size"],
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    _, _, test_dl = data_loader.get_dataloaders()
    
    fake_dataloader = FakeDataLoader(batch_size = config["fake_data"]["batch_size"], 
                                       num_workers = config["fake_data"]["num_workers"])
    fake_dl = fake_dataloader.get_dataloader()

    model_name = os.path.splitext(os.path.basename(config["model"]["path"]))[0]
    if "_" in model_name:
        model_name = model_name.split("_")[-1]
    
    #Creo il modello e carico i pesi
    model = create_model(model_name)
    model.to(device)
    model.load_state_dict(torch.load(config["model"]["path"], map_location=torch.device(device)))
    
    runner = Model_Runner(model)

    # Loop sulle temperature
    for temp in config["model"]["temps"]:
        print(f"\nTesting with temperature: {temp}")
        # Setup del logger per ogni temperatura
        logger = Logger(
            project_name=config["logging"]["project_name"],
            run_name=f"exp1_{model_name}_{temp}temp",
            config=config
        )
        runner.logger = logger

        #Test della CNN
        if model.type == "classifier":

            runner.setup()
    
            #Prendo le predizioni della CNN
            _, _, y_true, y_pred = runner.test(test_dl) 
    
            #Plot della matrice di confusione
            plot_confusion_matrix_accuracy(y_true, y_pred, test_dl, model_name, temp, save_path="plots/es1/")
    
            #Prendo il primo batch di immagini e label dal dataloader di test
            x, y = next(iter(test_dl))
            #Prendo il primo batch di immagini fake 
            x_fake, _ = next(iter(fake_dl))
    
            #Prendo la prima immagine del batch
            k = 0
            #Plot con i dati reali
            plot_logit_softmax(x, k, model, device, model_name, temp, save_path="plots/es1/", ty="real data") 
            #Plot con i dati fake
            plot_logit_softmax(x_fake, k, model, device, model_name, temp, save_path="plots/es1/", ty="fake data") 
    
            #Usando i logits
            scores_test = compute_scores(model, test_dl, max_logit, device)
            scores_fake = compute_scores(model, fake_dl, max_logit, device)
            plot_score(scores_test, scores_fake, model_name, temp, save_path="plots/es1/", score_fun="max_logit")
    
            #Usando la softmax
            scores_test = compute_scores(model, test_dl, lambda l: max_softmax(l, t = temp), device)
            scores_fake = compute_scores(model, fake_dl, lambda l: max_softmax(l, t = temp), device)
            plot_score(scores_test, scores_fake, model_name, temp, save_path="plots/es1/", score_fun="max_softmax")
    
        #Test dell'autoencoder
        else:
            criterion = config["train"]["criterion"]
            criterion_params = config["train"]["criterion_params"]
            runner.setup(criterion = criterion, criterion_params = criterion_params)
            _, _, test_score, _ = runner.test(test_dl)
            _, _, fake_score, _ = runner.test(fake_dl)
    
            plot_score(test_score, fake_score, model_name, temp, save_path="plots/es1/")
    
        logger.finish()

#Metodo dedicato per l'esercizio 2: Enhancing Robustness to Adversarial Attack
def experiment_2(config):
    
    print("Running Experiment 2: Enhancing Robustness to Adversarial Attack")
    
    #Device
    device = config["device"]
    
    #Carico i dati per training, validation e test
    data_loader = CIFAR10_DataLoader(
        batch_size=config["data"]["batch_size"],
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    train_dl, val_dl, test_dl = data_loader.get_dataloaders()
    
    #Carico anche i dati fake per la valutazione OOD
    fake_dataloader = FakeDataLoader(batch_size=config["data"]["batch_size"], 
                                   num_workers=config["data"]["num_workers"])
    fake_dl = fake_dataloader.get_dataloader()
    
    mean = config["data"]["mean"]
    std = config["data"]["std"]
    model_name = os.path.splitext(os.path.basename(config["model"]["path"]))[0]
    if "_" in model_name:
        model_name = model_name.split("_")[-1]

    #Creo il modello e carico i pesi
    model = create_model(model_name)
    model.to(device)
    model.load_state_dict(torch.load(config["model"]["path"], map_location=torch.device(device)))
    
    #Faccio il setup del logger
    logger = Logger(
        project_name=config["logging"]["project_name"],
        run_name=f"exp2_adv_train_{model_name}",
        config=config
    )
    
    #Test della CNN
    if model.type == "classifier":
        
        print("Step 1: Test del modello baseline")
        
        #Carico il modello pre-addestrato per il confronto baseline
        model_baseline = create_model(model_name)
        model_baseline.to(device)
        model_baseline.load_state_dict(torch.load(config["model"]["path"], map_location=device))
        
        #Creo il trainer per il baseline test
        trainer_baseline = FGSM_trainer(model_baseline, mean, std, logger=logger, device=device)
        trainer_baseline.setup_training(
            criterion=config["setup"]["criterion"], 
            criterion_params=config["setup"]["criterion_params"]
        )
        
        #Testo il modello baseline
        accuracies_baseline, examples_baseline = [], []
        epsilons = config["fgsm"]["epsilons_cnn"]
        
        tqdm_bar = tqdm(epsilons, total=len(epsilons), desc="[Testing Baseline Model]")
        for eps in tqdm_bar:
            test_acc, ex, y_true, y_pred = trainer_baseline.test_attack(test_dl, epsilon=eps)
            accuracies_baseline.append(test_acc[0] if test_acc[0] is not None else 0)
            examples_baseline.append(ex)
            tqdm_bar.set_postfix(epsilon=f"{eps}", test_accuracy=f"{test_acc[0]:.4f}")
        
        #Plot dei risultati baseline
        trainer_baseline.plot_result(epsilons, examples_baseline, accuracies_baseline, 
                                   f"{model_name}_baseline", save_path="plots/es2/")
        
        print("Step 2: Addestramento con augmentation  FGSM")
        
        #Creo un nuovo modello per l'addestramento 
        model_adv = create_model(model_name)
        model_adv.to(device)
        
        #Creo il trainer FGSM per l'addestramento 
        trainer_adv = FGSM_trainer(model_adv, mean, std, logger=logger, device=device)
        trainer_adv.setup_training(
            criterion=config["setup"]["criterion"], 
            criterion_params=config["setup"]["criterion_params"],
            optimizer=config["train"]["optimizer"],
            lr=config["train"]["lr"],
            scheduler=config["train"]["scheduler"],
            max_iter=config["train"]["epochs"]
        )
        
        #Addestramento con augmentation  FGSM on-the-fly
        clean_loss, adv_loss = trainer_adv.fgsm_train(
            dl_train=train_dl,
            train_epochs_num=config["train"]["epochs"],
            epsilon=config["fgsm"]["train_epsilon"]
        )
        
        print(f"Addestramento  completato - Clean Loss: {clean_loss:.4f}, Adv Loss: {adv_loss:.4f}")
        
        print("Step 3: Test del modello addestrato")
        
        #Testo il modello addestrato adversarialmente
        accuracies_adv, examples_adv = [], []
        
        tqdm_bar = tqdm(epsilons, total=len(epsilons), desc="[Testing Adversarially Trained Model]")
        for eps in tqdm_bar:
            test_acc, ex , y_true, y_pred = trainer_adv.test_attack(test_dl, epsilon=eps)
            accuracies_adv.append(test_acc[0] if test_acc[0] is not None else 0)
            examples_adv.append(ex)

            #Plot della matrice di confusione
            plot_confusion_matrix_accuracy(y_true, y_pred, test_dl, f"{model_name}_adv_trained", 
                                         "1.0", save_path="plots/es2/")
            
            #Score per OOD detection
            x, y = next(iter(test_dl))
            x_fake, _ = next(iter(fake_dl))
            k = 0
            
            #Plot di logits/softmax
            plot_logit_softmax(x, k, model_adv, device, f"{model_name}_adv_trained", "1.0", 
                              save_path="plots/es2/", ty="real_data_adv_trained")
            plot_logit_softmax(x_fake, k, model_adv, device, f"{model_name}_adv_trained", "1.0", 
                              save_path="plots/es2/", ty="fake_data_adv_trained")
            
            #Calcolo degli score OOD
            scores_test = compute_scores(model_adv, test_dl, max_logit, device)
            scores_fake = compute_scores(model_adv, fake_dl, max_logit, device)
            plot_score(scores_test, scores_fake, f"{model_name}_adv_trained", "1.0", 
                      save_path="plots/es2/", score_fun="max_logit")
            
            scores_test = compute_scores(model_adv, test_dl, lambda l: max_softmax(l, t=1.0), device)
            scores_fake = compute_scores(model_adv, fake_dl, lambda l: max_softmax(l, t=1.0), device)
            plot_score(scores_test, scores_fake, f"{model_name}_adv_trained", "1.0", 
                      save_path="plots/es2/", score_fun="max_softmax")

            
            tqdm_bar.set_postfix(epsilon=f"{eps}", test_accuracy=f"{test_acc[0]:.4f}")
            
        
        #Plot dei risultati del modello adversarialmente addestrato
        trainer_adv.plot_result(epsilons, examples_adv, accuracies_adv, 
                               f"{model_name}_adv_trained", save_path="plots/es2/")

          
        print(f"Accuracy a epsilon=0.0 - Baseline: {accuracies_baseline[0]:.4f}, Adv Trained: {accuracies_adv[0]:.4f}")
        if len(epsilons) > 1:
            print(f"Accuracy a epsilon={epsilons[1]} - Baseline: {accuracies_baseline[1]:.4f}, Adv Trained: {accuracies_adv[1]:.4f}")

    #Caso Autoencoder
    else:
        
        print("Step 1: Test dell'autoencoder baseline")
        
        #Carico il modello baseline
        model_baseline = create_model(model_name)
        model_baseline.to(device)
        model_baseline.load_state_dict(torch.load(config["model"]["path"], map_location=device))
        
        #Creo il trainer per il baseline
        trainer_baseline = FGSM_trainer(model_baseline, mean, std, logger=logger, device=device)
        trainer_baseline.setup_training(
            criterion=config["setup"]["criterion"], 
            criterion_params=config["setup"]["criterion_params"]
        )

        #Testo l'autoencoder baseline
        losses_baseline, examples_baseline = [], []
        epsilons = config["fgsm"]["epsilons_ae"]
        
        tqdm_bar = tqdm(epsilons, total=len(epsilons), desc="[Testing Baseline AE]")
        for eps in tqdm_bar:
            test_loss, ex, ae_scores, _ = trainer_baseline.test_attack(test_dl, epsilon=eps)
            losses_baseline.append(test_loss[0])
            examples_baseline.append(ex)
            tqdm_bar.set_postfix(epsilon=f"{eps}", avg_reconstruction_loss=f"{test_loss[0]:.4f}")
        
        #Plot dei risultati baseline
        trainer_baseline.plot_result(epsilons, examples_baseline, losses_baseline, 
                                   f"{model_name}_baseline", save_path="plots/es2/")
        
        print("Step 2: Addestramento  dell'autoencoder")
        
        #Creo nuovo modello per addestramento 
        model_adv = create_model(model_name)
        model_adv.to(device)
        
        #Creo il trainer per l'addestramento 
        trainer_adv = FGSM_trainer(model_adv, mean, std, logger=logger, device=device)
        trainer_adv.setup_training(
            criterion=config["setup"]["criterion"], 
            criterion_params=config["setup"]["criterion_params"],
            optimizer=config["train"]["optimizer"],
            lr=config["train"]["lr"],
            scheduler=config["train"]["scheduler"],
            max_iter=config["train"]["epochs"]
        )
        
        #Addestramento con augmentation FGSM
        clean_loss, adv_loss = trainer_adv.fgsm_train(
            dl_train=train_dl,
            train_epochs_num=config["train"]["epochs"],
            epsilon=config["fgsm"]["train_epsilon"]
        )
        
        print(f"Addestramento  completato - Clean Loss: {clean_loss:.4f}, Adv Loss: {adv_loss:.4f}")
        
        print("Step 3: Test dell'autoencoder addestrato")
        
        #Testo l'autoencoder addestrato adversarialmente
        losses_adv, examples_adv = [], []
        
        tqdm_bar = tqdm(epsilons, total=len(epsilons), desc="[Testing Adversarially Trained AE]")
        for eps in tqdm_bar:
            test_loss, _, ae_scores, _, _, ex = trainer_adv.test_attack(test_dl, epsilon=eps)
            losses_adv.append(test_loss[0])
            examples_adv.append(ex)

            #Prendo gli score sui dati di test e fake
            _, _, test_score, _ = runner.test(test_dl)
            _, _, fake_score, _ = runner.test(fake_dl)
            
            #Plot degli score per OOD detection
            plot_score(test_score, fake_score, f"{model_name}_adv_trained", "1.0", save_path="plots/es2/")
            
            tqdm_bar.set_postfix(epsilon=f"{eps}", avg_reconstruction_loss=f"{test_loss[0]:.4f}")
        
        #Plot dei risultati 
        trainer_adv.plot_result(epsilons, examples_adv, losses_adv, 
                               f"{model_name}_adv_trained", save_path="plots/es2/")
        
        print(f"Loss a epsilon=0.0 - Baseline: {losses_baseline[0]:.4f}, Adv Trained: {losses_adv[0]:.4f}")
        if len(epsilons) > 1:
            print(f"Loss a epsilon={epsilons[1]} - Baseline: {losses_baseline[1]:.4f}, Adv Trained: {losses_adv[1]:.4f}")
            
        
        
    print(f"Tutti i plot salvati in plots/es2/ - controlla {model_name}_baseline vs {model_name}_adv_trained")
    logger.finish()

#Metodo dedicato per l'esercizio 3: Implement JARN
def experiment_3(config):

    print("Running Experiment 3: Implement JARN")

    #Device
    device = config["device"]

    #Carico i dati
    data_loader = CIFAR10_DataLoader(
        batch_size=128,
        split=20,
        num_workers=4
    )
    train_dl, _, test_dl = data_loader.get_dataloaders()

    mean = config["data"]["mean"]
    std = config["data"]["std"]


    classifier_name = "CNNplus"
    #Creo il classificatore e il discriminatore
    classifier = create_model(classifier_name)
    classifier.to(device)
    
    discriminator = create_model("CNNdiscriminator")
    discriminator.to(device)
    
    #Logger
    logger = Logger(
        project_name=config["logging"]["project_name"],
        run_name=f"exp3_JARN_{classifier_name}",
        config=config
    )

    #Creo trainer FGSM
    trainer = FGSM_trainer(classifier, mean, std, logger=logger)

    trainer.setup_training(
        criterion="CrossEntropyLoss",
        optimizer="Adam",
        lr=config["training"]["lr"],
        scheduler="CosineAnnealingLR",
        max_iter=config["training"]["epochs"]
    )

    #Training JARN
    print("Inizio addestramento con JARN...")
    trainer.jarn_train(
        dl_train=train_dl,
        train_epochs_num=config["training"]["epochs"],
        discriminator=discriminator,
        adv_weight=config["jarn"]["adv_weight"],
        epsilon=config["jarn"]["epsilon"],
        disc_opt_lr=config["jarn"]["disc_opt_lr"],
        jarn_start_ratio=config["jarn"]["start_ratio"]
    )
    print("Training with JARN terminated")

    #Carico i dati
    data_loader = CIFAR10_DataLoader(
        batch_size=1,
        split=20,
        num_workers=4
    )
    _, _, test_dl = data_loader.get_dataloaders()

    #Test con attacchi FGSM
    accuracies, examples = [], []
    epsilons = config["fgsm"]["epsilons"]

    tqdm_bar = tqdm(epsilons, total=len(epsilons), desc="[JARN Testing Epsilon]")
    for eps in tqdm_bar:
        acc, ex, _, _ = trainer.test_attack(test_dl, epsilon=eps)
        accuracies.append(acc)
        examples.append(ex)
        tqdm_bar.set_postfix(epsilon=f"{eps}", test_accuracy=f"{acc:.4f}")

    #Plot risultati
    trainer.plot_result(
        epsilons, examples, accuracies, "JARN_CNN_plus",
        save_path="plots/es3/"
    )
    print("All plots saved in plots/es3/")

    logger.finish()


def main():
    
    parser = argparse.ArgumentParser(description="Run OOD experiments")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["1", "2", "3", "pretrain"],
                       help="Which experiment to run")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    #Si setta il device scelto
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    torch.cuda.empty_cache() if device == "cuda" else None

    config_path = f"./configs/config_{args.experiment}.yaml"
    config = load_config(config_path)
    
    #Si caricano le configurazioni
    config = load_config(config_path)
    config["device"] = device

    if "seed" in config:
        set_seed(config["seed"])
    else:
        print("Warning: Nessun seed specificato nella configurazione")
    
    #Si fa partire l"esperimento
    experiment_map = {
        "1": experiment_1,
        "2": experiment_2,
        "3": experiment_3,
        "pretrain" : pretrain
    }
    
    experiment_fn = experiment_map[args.experiment]
    experiment_fn(config)

if __name__ == "__main__":
    main()