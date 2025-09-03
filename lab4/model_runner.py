import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

#Classe per il pre-training se necessario ripresa dal lab 1
#sono state apportate alcune modifiche per la gestione dell'autoencoder
#e nella funzione di test sempre per l'esercizio 1
class Model_Runner:
    
    def __init__(self, model, logger=None):

        #Inizializzo il modello e lo sposto su device utilizzato
        self.model = model
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self.device)

        #Preparo i campi che saranno inizializzati col setup
        self.criterion = None
        self.model_type = None
        self.lr = None
        self.optimizer = None
        self.scheduler = None
        
        
    #Metodo che definisce le specifiche dell'esperimento, aggiunta la possibilità di passare
    #parametri per le loss, utile quando dovrò lavorare con l'autoencoder
    def setup(self, criterion="CrossEntropyLoss", criterion_params=None,
                       optimizer="Adam", lr=0.001, scheduler="CosineAnnealingLR", max_iter=100):

        #Mi serve a garantire che sia sempre un dizionario
        #per non avere errori quando passo **criterion_params
        criterion_params = criterion_params or {}
        self.criterion = getattr(nn, criterion)(**criterion_params)
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr)
        self.scheduler = getattr(lr_scheduler, scheduler)(self.optimizer, max_iter)
        self.lr = lr

    #Metodo che gestisce l'addestramento del modello
    def train(self, dl_train, dl_val, train_epochs_num):

        train_losses = []
        train_accuracies = []
        validation_losses = []
        validation_accuracies = []

        self.model.to(self.device)
    
        for epoch in range(train_epochs_num):
                
            #Reset delle statistiche per ogni epoca
            running_loss, correct, total = 0.0, 0, 0
    
            #Fase di addestramento
            self.model.train()
    
            for inputs, targets in tqdm(dl_train, desc=f"Training Epoch {epoch+1}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logits = self.model(inputs) 
                if self.model.type == "classifier":
                    loss = self.criterion(logits, targets)
                else:  # autoencoder
                    _, xr = self.model(inputs)
                    loss = self.criterion(xr, inputs).mean()
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                #Aggiornamento delle statistiche    
                running_loss += loss.item() * inputs.size(0)                    
                if self.model.type == "classifier":
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                else:
                    total += inputs.size(0)
    
            train_loss = running_loss / total
            train_losses.append(train_loss)
            train_acc = (correct / total) if self.model.type == "classifier" else None
            train_accuracies.append(train_acc)
                
            #Fase di validazione
            val_loss, val_acc = self.validate(dl_val)
            validation_losses.append(val_loss)
            validation_accuracies.append(val_acc)
    
            #Logging metriche su wandb
            if self.logger:
                metrics = {
                    "train/loss": train_loss, 
                    "val/loss": val_loss,
                }
                if self.model.type == "classifier":
                    metrics.update({"train/accuracy": train_acc, "val/accuracy": val_acc})
                self.logger.log_metrics(metrics, step=epoch)
    
            #Aggiornamento dello scheduler
            self.scheduler.step()
    
        return train_losses, train_accuracies, validation_losses, validation_accuracies

    #Metodo che gestisce la validazione del modello
    def validate(self, dl_val):

        #Fase di valutazione
        self.model.to(self.device)
        self.model.eval()
        
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(dl_val, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
        
                logits = self.model(inputs)
                if self.model.type == "classifier":
                    loss = self.criterion(logits, targets)
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                else:  # autoencoder
                    _, xr = self.model(inputs)
                    loss = self.criterion(xr, inputs).mean()
                    total += inputs.size(0)
    
                #Aggiornamento delle statistiche
                running_loss += loss.item() * inputs.size(0)
                    
        val_loss = running_loss / total
        val_acc = (correct / total) if self.model.type == "classifier" else None
    
        return val_loss, val_acc

    #Metodo che gestisce il testing sui modelli
    def test(self, dl_test):

        #Fase di testing
        self.model.to(self.device)
        self.model.eval()

        running_loss, correct, total = 0.0, 0, 0
        y_true = []
        y_pred = []
        ae_scores = []
       
        with torch.no_grad():
            for inputs, targets in tqdm(dl_test, desc="Testing"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                logits = self.model(inputs)
                if self.model.type == "classifier":
                    loss = self.criterion(logits, targets)
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    y_true.append(targets.view(-1).cpu())
                    y_pred.append(preds.view(-1).cpu())
                    
                else:  #Autoencoder
                    z, xr = self.model(inputs)
                    l = self.criterion(xr, inputs)
                    #Per-sample MSE
                    score = l.mean(dim=[1, 2, 3])
                    #Negativo per OOD score
                    ae_scores.append(-score)  
                    loss = l.mean()
                    total += inputs.size(0)

                running_loss += loss.item() * inputs.size(0)

        test_loss = running_loss / total
        test_acc = (correct / total) if self.model.type == "classifier" else None
        
        #Logging su wandb
        if self.logger:
            metrics = {"test/loss": test_loss}
            if self.model.type == "classifier":
                metrics["test/accuracy"] = test_acc
            self.logger.log_metrics(metrics)
    
        #Ritorno valori coerenti
        if self.model.type == "classifier":
            return [test_loss], [test_acc], y_true, y_pred
        else:
            ae_scores = torch.cat(ae_scores)
            return [test_loss], [None], ae_scores, None# Lista per salvare esempi avversari
        total_loss = 0      # Solo per AE
        correct = 0         # Solo per CNN
        y_true = []
        y_pred = []
        ae_scores = []
    
        tqdm_bar = tqdm(test_loader, total=len(test_loader),
                        desc=f"[FGSM attack epsilon:{epsilon}]", leave=False)
        for data, target in tqdm_bar:
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
    
            #  Forward originale 
            if self.model.type == "classifier":
                output = self.model(data)
                loss = self.criterion(output, target)
                init_pred = output.max(1, keepdim=True)[1]
                if init_pred.item() != target.item():
                    continue
            else:  # Autoencoder
                _, output = self.model(data)
                loss = self.criterion(output, data).mean()
    
            #  Gradiente e perturbazione 
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            data_denorm = denorm(data, self.mean, self.std, self.device)
            perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
            perturbed_data_normalized = transforms.Normalize(self.mean, self.std)(perturbed_data)
    
            #  Forward con immagine perturbata 
            if self.model.type == "classifier":
                output = self.model(perturbed_data_normalized)
                loss = self.criterion(output, target).mean()
                final_pred = output.max(1)[1]
    
                # Metriche
                correct += (final_pred == target).sum().item()
                y_true.append(target.view(-1).cpu())
                y_pred.append(final_pred.view(-1).cpu())
    
                # Salvo esempi
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data[0].squeeze().detach().cpu().numpy()
                    adv_examples.append((adv_ex, loss.item()))
    
            else:  # Autoencoder
                _, output = self.model(perturbed_data_normalized)
                loss_adv = self.criterion(output, data).mean()
                total_loss += loss_adv.item()
    
                # Per-sample MSE per OOD
                score = self.criterion(output, data).mean(dim=[1, 2, 3])
                ae_scores.append(-score)
    
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data[0].squeeze().detach().cpu().numpy()
                    adv_examples.append((adv_ex, loss_adv.item()))
    
        #  Risultati finali 
        if self.model.type == "classifier":
            final_acc = correct / float(len(test_loader))
            print(f"Epsilon: {epsilon}\nTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    
            if self.logger:
                self.logger.log_metrics({f"FGSM/test_acc_eps_{epsilon}": final_acc})
    
            return final_acc, adv_examples, y_true, y_pred
    
        else:
            avg_loss = total_loss / len(test_loader)
            print(f"Epsilon: {epsilon}\nAvg Reconstruction Loss = {avg_loss}")
    
            if self.logger:
                self.logger.log_metrics({f"FGSM/test_loss_eps_{epsilon}": avg_loss})
    
            ae_scores = torch.cat(ae_scores)
            return avg_loss, adv_examples, ae_scores, None
