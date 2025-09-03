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

# Funzione per tutti i plot comuni (matrice confusione, logits/softmax, score)
def plot_all(model, device, test_dl, fake_dl, model_name, temp, save_path="plots/"):
    # Primo batch dati reali
    x_real, y_real = next(iter(test_dl))
    x_fake, _ = next(iter(fake_dl))
    k = 0

    # Plot logits/softmax per dati reali e fake
    plot_logit_softmax(x_real, k, model, device, model_name, temp, save_path=save_path, ty="real_data")
    plot_logit_softmax(x_fake, k, model, device, model_name, temp, save_path=save_path, ty="fake_data")

    # Score OOD usando logits
    scores_test = compute_scores(model, test_dl, max_logit, device)
    scores_fake = compute_scores(model, fake_dl, max_logit, device)
    plot_score(scores_test, scores_fake, model_name, temp, save_path=save_path, score_fun="max_logit")

    # Score OOD usando softmax
    scores_test = compute_scores(model, test_dl, lambda l: max_softmax(l, t=temp), device)
    scores_fake = compute_scores(model, fake_dl, lambda l: max_softmax(l, t=temp), device)
    plot_score(scores_test, scores_fake, model_name, temp, save_path=save_path, score_fun="max_softmax")



def test_AE(model, dataloader, device):
    model.eval()
    loss = nn.MSELoss(reduction='none')
    scores = []
    losses = []
    tqdm_bar = tqdm(dataloader, desc="[Testing (Val/Test/Fake)]", leave=False)
    with torch.no_grad():
        for data in tqdm_bar:
            x, y = data
            x = x.to(device)
            z, xr = model(x)
            l = loss(x, xr)
            score = l.mean([1, 2, 3])

            losses.append(score)
            scores.append(-score)

    scores = torch.cat(scores)
    losses = torch.mean(torch.cat(losses))
    return  scores, losses.item()

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
    
    device = config["device"]

    # Carico i dataloader
    data_loader = CIFAR10_DataLoader(
        batch_size=config["data"]["batch_size"],
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    _, _, test_dl = data_loader.get_dataloaders()
    
    fake_dl = FakeDataLoader(
        batch_size=config["fake_data"]["batch_size"],
        num_workers=config["fake_data"]["num_workers"]
    ).get_dataloader()
    
    # Determino modello
    model_name = os.path.splitext(os.path.basename(config["model"]["path"]))[0].split("_")[-1]
    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(config["model"]["path"], map_location=device))

    temp_list = config["model"]["temps"]

    for temp in temp_list:
        print(f"\nTesting with temperature: {temp}")

        # --- CNN o classificatore ---
        if model.type == "classifier":
            y_true, y_pred = [], []
            for data, target in test_dl:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.max(1)[1]

                y_true.append(target)
                y_pred.append(pred)

            # Plot matrice di confusione
            plot_confusion_matrix_accuracy(y_true, y_pred, test_dl, model_name, temp, save_path="plots/es1/")

            # Plot logits/softmax reali e fake
            x_real, _ = next(iter(test_dl))
            x_fake, _ = next(iter(fake_dl))
            k = 0
            plot_logit_softmax(x_real, k, model, device, model_name, temp, save_path="plots/es1/", ty="real_data")
            plot_logit_softmax(x_fake, k, model, device, model_name, temp, save_path="plots/es1/", ty="fake_data")

            # Score OOD con logit
            scores_test = compute_scores(model, test_dl, max_logit, device).flatten().detach()
            scores_fake = compute_scores(model, fake_dl, max_logit, device).flatten().detach()
            plot_score(scores_test, scores_fake, model_name, temp, save_path="plots/es1/", score_fun="max_logit")

            # Score OOD con softmax
            scores_test = compute_scores(model, test_dl, lambda l: max_softmax(l, t=temp), device).flatten().detach()
            scores_fake = compute_scores(model, fake_dl, lambda l: max_softmax(l, t=temp), device).flatten().detach()
            plot_score(scores_test, scores_fake, model_name, temp, save_path="plots/es1/", score_fun="max_softmax")

        # --- Autoencoder ---
        else:
            
            # Autoencoder: score OOD
            test_score, _ = test_AE(model, test_dl, device)
            fake_score, _ = test_AE(model, fake_dl, device)
            
            # Converto in NumPy per plot_score
            plot_score(test_score, fake_score, model_name, temp, save_path="plots/es1/")



#Metodo dedicato per l'esercizio 2: Enhancing Robustness to Adversarial Attack
def experiment_2(config):
    print("Running Experiment 2: Enhancing Robustness to Adversarial Attack")

    device = config["device"]

    # Dataloaders train/val/test
    data_loader = CIFAR10_DataLoader(
        batch_size=config["data"]["batch_size"],
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    train_dl, val_dl, test_dl = data_loader.get_dataloaders()
    _, _, at_test_dl = CIFAR10_DataLoader(batch_size=1, split=20, num_workers=0).get_dataloaders()
    fake_dl = FakeDataLoader(batch_size=config["data"]["batch_size"], num_workers=config["data"]["num_workers"]).get_dataloader()

    mean, std = config["data"]["mean"], config["data"]["std"]
    model_name = os.path.splitext(os.path.basename(config["model"]["path"]))[0].split("_")[-1]

    # Creo modello baseline e carico pesi
    model_baseline = create_model(model_name)
    model_baseline.to(device)
    model_baseline.load_state_dict(torch.load(config["model"]["path"], map_location=device))

    # Logger
    logger = Logger(project_name=config["logging"]["project_name"], run_name=f"exp2_{model_name}", config=config)

    # Creo trainer FGSM per baseline e adv training
    trainer_baseline = FGSM_trainer(model_baseline, mean, std, logger=logger, device=device)
    trainer_baseline.setup_training(
        criterion=config["setup"]["criterion"],
        criterion_params=config["setup"]["criterion_params"],
        optimizer=config["setup"].get("optimizer", None),
        lr=config["setup"].get("lr", None),
        scheduler=config["setup"].get("scheduler", None),
        max_iter=config["setup"].get("epochs", None)
    )

    # Determino epsilon in base al tipo di modello
    if model_baseline.type == "classifier":
        epsilons = config["fgsm"]["epsilons_cnn"]
        train_func = trainer_baseline.train_classifier
        test_func = trainer_baseline.test_classifier
    else:
        epsilons = config["fgsm"]["epsilons_ae"]
        train_func = trainer_baseline.train_autoencoder
        test_func = trainer_baseline.test_autoencoder

    print("Step 1: Test del modello baseline")
    results_baseline = []

    for eps in tqdm(epsilons, desc="[Testing Baseline Model]"):
        res = test_func(at_test_dl, fgsm=True, epsilon=eps)
        results_baseline.append(res)

    # Plot dei risultati baseline
    if model_baseline.type == "classifier":
        # CNN: plot confusion matrix e logits/softmax
        y_true = res[2]
        y_pred = res[3]
        plot_confusion_matrix_accuracy(y_true, y_pred, test_dl, f"{model_name}_baseline", "1.0", save_path="plots/es2/")
        plot_all(model_baseline, device, test_dl, fake_dl, f"{model_name}_baseline", temp=1.0, save_path="plots/es2/")
    else:
        # AE: plot OOD score
        _, _, test_score, _ = test_func(test_dl, fgsm=False)
        _, _, fake_score, _ = test_func(fake_dl, fgsm=False)
        test_score = test_score.detach().cpu()
        fake_score = fake_score.detach().cpu()
        plot_score(test_score, fake_score, f"{model_name}_baseline", 1.0, save_path="plots/es2/")

    print("Step 2: Addestramento con FGSM")
    model_adv = create_model(model_name)
    model_adv.to(device)

    trainer_adv = FGSM_trainer(model_adv, mean, std, logger=logger, device=device)
    trainer_adv.setup_training(
        criterion=config["setup"]["criterion"],
        criterion_params=config["setup"]["criterion_params"],
        optimizer=config["setup"]["optimizer"],
        lr=config["setup"]["lr"],
        scheduler=config["setup"]["scheduler"],
        max_iter=config["setup"]["epochs"]
    )

    clean_loss, adv_loss = train_func(train_dl, train_epochs_num=config["setup"]["epochs"], fgsm=True, epsilon=config["fgsm"]["train_epsilon"])
    print(f"Addestramento completato - Clean Loss: {clean_loss:.4f}, Adv Loss: {adv_loss:.4f}")

    print("Step 3: Test modello addestrato")
    results_adv = []

    for eps in tqdm(epsilons, desc="[Testing Adversarially Trained Model]"):
        res = test_func(at_test_dl, fgsm=True, epsilon=eps)
        results_adv.append(res)

    # Plot finale
    if model_baseline.type == "classifier":
        y_true = res[2]
        y_pred = res[3]
        plot_confusion_matrix_accuracy(y_true, y_pred, test_dl, f"{model_name}_adv_trained", "1.0", save_path="plots/es2/")
        plot_all(model_adv, device, test_dl, fake_dl, f"{model_name}_adv_trained", temp=1.0, save_path="plots/es2/")
    else:
        _, _, test_score, _ = test_func(test_dl, fgsm=False)
        _, _, fake_score, _ = test_func(fake_dl, fgsm=False)
        test_score = test_score.detach().cpu()
        fake_score = fake_score.detach().cpu()
        plot_score(test_score, fake_score, f"{model_name}_adv_trained", 1.0, save_path="plots/es2/")

    trainer_adv.plot_result(epsilons, [r[1] for r in results_adv],
                            [r[0] for r in results_adv], f"{model_name}_adv_trained", save_path="plots/es2/")

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
        acc, ex, _, _ = trainer.test_classifier(test_dl, fgsm=True, epsilon=eps)
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
