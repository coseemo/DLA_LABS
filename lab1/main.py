import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloaders import MNIST_DataLoader, CIFAR10_DataLoader, Distillation_Dataset
from logger import Logger
from runners import Model_Runner, Distillery_Runner
from models import MLP, CNN
import os
import sys
import random
import numpy as np


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
def create_model(model_config):
    model_type = model_config["type"]
    params = model_config["params"]
    
    if model_type == "MLP":
        model = MLP(**params)
    elif model_type == "CNN":
        model = CNN(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

#Metodo dedicato per l'esercizio 1.1: MLP con MNIST
def experiment_1_1(config):
    print("Running Experiment 1.1: Baseline MLP on MNIST")
    
    #Carico i dati
    data_loader = MNIST_DataLoader(
        batch_size=config["data"]["batch_size"],
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    train_dl, val_dl, test_dl = data_loader.get_dataloaders()
    
    #Creo il modello
    model = create_model(config["model"])
    
    #Faccio il setup del logger
    logger = Logger(
        project_name=config["logging"]["project_name"],
        run_name=f"exp1.1_{config["model"]["params"]["hidden_layers_num"]}layers",
        config=config
    )
    
    runner = Model_Runner(model, logger)
    
    #Imposto l"allenamento
    runner.setup_training(
        criterion=config["training"]["criterion"],
        optimizer=config["training"]["optimizer"],
        lr=config["training"]["lr"],
        scheduler=config["training"]["scheduler"],
        max_iter=config["training"]["epochs"]
    )
    
    #Addestramento
    train_losses, train_accs, val_losses, val_accs = runner.train(
        train_dl, val_dl, config["training"]["epochs"]
    )
    
    #Testing
    test_losses, test_accs = runner.test(test_dl, 1)
    
    print(f"Final test accuracy: {test_accs[0]:.4f}")

    if logger:
        logger.log_metrics({
            "test/loss": test_losses[0],
            "test/accuracy": test_accs[0]
        })
    
    logger.finish()

#Metodo dedicato per l'esercizio 1.2: MLP con e senza connessioni residuali
def experiment_1_2(config):
    
    print("Running Experiment 1.2: MLP with residual connections")
    
    #Carico i dati
    data_loader = MNIST_DataLoader(
        batch_size=config["data"]["batch_size"],
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    train_dl, val_dl, test_dl = data_loader.get_dataloaders()
    
    results = {}

    for width in config["experiment"]["widths"]:
        for depth in config["experiment"]["depths"]:
            for residual in [False, True]:
                model_config = config["model"].copy()
                model_config["params"]["hidden_layers_num"] = depth
                model_config["params"]["residual"] = residual
                
                model = create_model(model_config)
                
                #Setup del logger
                logger = Logger(
                    project_name=config["logging"]["project_name"],
                    run_name=f"exp1.2_d{depth}_res{residual}",
                    config={**config, "depth": depth, "residual": residual}
                )
                
                #Importo il runner
                runner = Model_Runner(model, logger)
                
                #Setup dell'addestramento
                runner.setup_training(
                    criterion=config["training"]["criterion"],
                    optimizer=config["training"]["optimizer"],
                    lr=config["training"]["lr"],
                    scheduler=config["training"]["scheduler"],
                    max_iter=config["training"]["epochs"]
                )
                
                #Addestramento
                train_losses, train_accs, val_losses, val_accs = runner.train(
                    train_dl, val_dl, config["training"]["epochs"]
                )
                
                #Testing
                test_losses, test_accs = runner.test(test_dl, 1)
                
                #Analisi dei gradienti
                if config["experiment"].get("analyze_gradients", False):
                    runner.gradient_norm(train_dl)
                
                results[f"depth_{depth}_residual_{residual}"] = {
                    "test_accuracy": test_accs[0],
                    "final_val_loss": val_losses[-1]
                }
                
                print(f"Depth {depth}, Residual {residual}: Test Acc = {test_accs[0]:.4f}")
                logger.finish()
        
    #Stampa i risultati ottenuti:
    #confronta MLP con e senza connessioni residuali a profondità diverse
    print("\nResults Summary:")
    for key, value in results.items():
        print(f"{key}: {value}")

#Metodo dedicato per l'esercizio 1.3: CNN con e senza connessioni residuali
def experiment_1_3(config):
    
    print("Running Experiment 1.3: CNN with residual connections on CIFAR-10")
    
    #Stavolta carichiamo CIFAR10
    data_loader = CIFAR10_DataLoader(
        batch_size=config["data"]["batch_size"],
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    train_dl, val_dl, test_dl = data_loader.get_dataloaders()
    
    results = {}
    
    for depth_list in config["experiment"]["depths"]:
        for residual in [False, True]:
            model_config = config["model"].copy()
            model_config["params"]["layers"] = depth_list
            model_config["params"]["residual"] = residual
            
            model = create_model(model_config)
            
            #Setup del logger
            logger = Logger(
                project_name=config["logging"]["project_name"],
                run_name=f"exp1.3_d{depth}_res{residual}",
                config={**config, "depth": depth, "residual": residual}
            )
        
            runner = Model_Runner(model, logger)
            
            #Setup del training
            runner.setup_training(
                criterion=config["training"]["criterion"],
                optimizer=config["training"]["optimizer"],
                lr=config["training"]["lr"],
                scheduler=config["training"]["scheduler"],
                max_iter=config["training"]["epochs"]
            )
            
            #Addestramento
            train_losses, train_accs, val_losses, val_accs = runner.train(
                train_dl, val_dl, config["training"]["epochs"]
            )
            
            #Testing
            test_losses, test_accs = runner.test(test_dl, 1)
            
            results[f"depth_{depth}_residual_{residual}"] = {
                "test_accuracy": test_accs[0],
                "final_val_loss": val_losses[-1],
                "model_params": sum(p.numel() for p in model.parameters())
            }
            
            print(f"Depth {depth}, Residual {residual}: Test Acc = {test_accs[0]:.4f}")
            logger.finish()
    
    #Stampiamo un confronto identico a quello svolto precedentemente per l"MLP
    #ma stavolta utilizziamo una CNN
    print("\nResults Summary:")
    for key, value in results.items():
        print(f"{key}: {value}")

#Metodo dedicato per l"esercizio 2.2: Knowledge Distillation
def experiment_2_2(config):

    print("Running Experiment 2.2: Knowledge Distillation")
    
    #Carico CIFAR10
    data_loader = CIFAR10_DataLoader(
        batch_size=config["data"]["batch_size"],
        split=config["data"]["validation_split"],
        num_workers=config["data"]["num_workers"]
    )
    train_dl, val_dl, test_dl = data_loader.get_dataloaders() 
    
    #Step 1: Alleniamo il modello teacher
    print("Step 1: Training teacher model...")
    teacher_model = create_model(config["teacher_model"])
    
    teacher_logger = Logger(
        project_name=config["logging"]["project_name"],
        run_name="teacher_model",
        config=config
    )
    
    teacher_runner = Model_Runner(teacher_model, teacher_logger)
    teacher_runner.setup_training(
        criterion=config["training"]["criterion"],
        optimizer=config["training"]["optimizer"],
        lr=config["training"]["lr"],
        scheduler=config["training"]["scheduler"]
    )
    
    #Osserviamo le performance del teacher
    teacher_runner.train(train_dl, val_dl, config["training"]["epochs"])
    teacher_test_losses, teacher_test_accs = teacher_runner.test(test_dl, 1)
    print(f"Teacher test accuracy: {teacher_test_accs[0]:.4f}")
    teacher_logger.finish()
    
    #Step 2: Alleniamo lo studente
    print("Step 2: Training student model (baseline)...")
    student_model = create_model(config["student_model"])
    
    student_logger = Logger(
        project_name=config["logging"]["project_name"],
        run_name="student_baseline",
        config=config
    )
    
    student_runner = Model_Runner(student_model, student_logger)
    student_runner.setup_training(
        criterion=config["training"]["criterion"],
        optimizer=config["training"]["optimizer"],
        lr=config["training"]["lr"],
        scheduler=config["training"]["scheduler"],
        max_iter=config["training"]["epochs"]
    )
    
    #Osserviamo le performance dello studente senza distillazione
    student_runner.train(train_dl, val_dl, config["training"]["epochs"])
    student_test_losses, student_test_accs = student_runner.test(test_dl, 1)
    print(f"Student baseline test accuracy: {student_test_accs[0]:.4f}")
    student_logger.finish()
    
    #Step 3: Knowledge Distillation
    print("Step 3: Knowledge Distillation...")
    
    #Creaiamo un nuovo studente per confrontarlo con quello senza distillazione
    distilled_student = create_model(config["student_model"])
    
    distill_logger = Logger(
        project_name=config["logging"]["project_name"],
        run_name="distilled_student",
        config=config
    )
    
    distillery = Distillery_Runner(
        student=distilled_student,
        teacher=teacher_model,
        logger=distill_logger,
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"]
    )
    
    distillery.setup_optimizer(
        optimizer_class=getattr(optim, config["training"]["optimizer"]),
        lr=config["training"]["lr"]
    )

    #Creazione del dataset per la distillazione
    #Senza shuffle solo per facilitare la corrispondenza 
    #tra indici del dataset originale e previsioni del teacher
    distill_train_dl, _, _ = data_loader.get_dataloaders(shuffle=False)  
    teacher_logits = distillery.obtain_teacher_logits(distill_train_dl)
    distill_dl, _, _ = data_loader.create_distillation_dataloader(self, original_data, teacher_logits)
    
    #Attuiamo la distillazione
    distill_losses, distill_accs = distillery.distillation(
        distill_dl, config["training"]["epochs"]
    )
    
    #Testiamo le performance dello studente con distillazione
    distilled_runner = Model_Runner(distilled_student, distill_logger)
    distilled_runner.setup_training(
        criterion=config["training"]["criterion"],
        optimizer=config["training"]["optimizer"],
        lr=config["training"]["lr"],
        scheduler=config["training"]["scheduler"],
        max_iter=config["training"]["epochs"]
    )
    
    distilled_test_losses, distilled_test_accs = distilled_runner.test(test_dl, 1)

    #Confronto delle performance tra i vari modelli
    print(f"\nDistillation Results:")
    print(f"Teacher accuracy: {teacher_test_accs[0]:.4f}")
    print(f"Student baseline accuracy: {student_test_accs[0]:.4f}")
    print(f"Distilled student accuracy: {distilled_test_accs[0]:.4f}")
    
    #Confronto tra il numero di parametri dei modelli e riduzione percentuale
    #(.numel() restituisce il numero totale di elementi all'interno del tensore)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in distilled_student.parameters())
    
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Parameter reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    distill_logger.finish()

def main():
    
    parser = argparse.ArgumentParser(description="Run CNN/MLP experiments")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["1_1", "1_2", "1_3", "2_2"],
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

    config_path = f"./configs/config_exp{args.experiment}.yaml"
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
        "1_1": experiment_1_1,
        "1_2": experiment_1_2,
        "1_3": experiment_1_3,
        "2_2": experiment_2_2
    }
    
    experiment_fn = experiment_map[args.experiment]
    experiment_fn(config)

if __name__ == "__main__":
    main()
