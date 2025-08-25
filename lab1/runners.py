import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb

#Classe che gestisce le fasi di training e testing dei singoli modelli
class Model_Runner:
    
    def __init__(self, model, logger=None, device = "cuda" if torch.cuda.is_available() else "cpu"):

        #Inizializzo il modello e lo sposto su device utilizzato
        self.model = model
        self.device = device
        model.to(device)

        #Inizializzo il logger
        self.logger = logger

        #Preparo i campi che saranno inizializzati col setup
        self.lr = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    #Metodo che definisce le specifiche dell'esperimento
    def setup_training(self, criterion="CrossEntropyLoss", optimizer="Adam", lr=0.001, scheduler="CosineAnnealingLR", max_iter=100):

        #Inizializza la loss, l'ottimizzatore e lo scheduler
        self.criterion = getattr(nn, criterion)()
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr)
        self.scheduler = getattr(lr_scheduler, scheduler)(self.optimizer, max_iter)
        

    #Metodo che gestisce l'addestramento del modello
    def train(self, dl_train, dl_val, train_epochs_num):

        train_losses = []
        train_accuracies = []
        validation_losses = []
        validation_accuracies = []
    
        for epoch in range(train_epochs_num):
                
                #Reset delle statistiche per ogni epoca
                running_loss, correct, total = 0.0, 0, 0
    
                #Fase di addestramento
                self.model.train()
    
                for inputs, targets in tqdm(dl_train, desc=f"Training Epoch {epoch+1}"):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
        
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)
    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
    
                    #Aggiornamento delle statistiche    
                    running_loss += loss.item() * inputs.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
    
                train_loss = running_loss / total
                train_acc = correct / total
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
    
                #Fase di validazione
                val_loss, val_acc = self.validate(dl_val)
                validation_losses.append(val_loss)
                validation_accuracies.append(val_acc)
    
                #Logging metriche su wandb
                if self.logger:
                    self.logger.log_metrics({
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                    }, step=epoch)
    
                #Aggiornamento dello scheduler
                self.scheduler.step()
    
        return train_losses, train_accuracies, validation_losses, validation_accuracies

    #Metodo che gestisce la validazione del modello
    def validate(self, dl_val):

            #Fase di valutazione
            self.model.eval()
        
            running_loss, correct, total = 0.0, 0, 0
            
            with torch.no_grad():
                for inputs, targets in tqdm(dl_val, desc="Validation"):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
        
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)
    
                    #Aggiornamento delle statistiche
                    running_loss += loss.item() * inputs.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
    
            return running_loss / total, correct / total

    
    #Metodo che gestisce il testing sui modelli
    def test(self, dl_test, test_epochs_num):

        #Fase di testing
        self.model.eval()

        running_loss, correct, total = 0.0, 0, 0
        
        test_losses = []
        test_accuracies = []

        for _ in range(test_epochs_num):
            with torch.no_grad():
                for inputs, targets in tqdm(dl_test, desc="Testing"):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
        
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)

                    #Aggiornamento delle statistiche
                    running_loss += loss.item() * inputs.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

            test_loss = running_loss / total
            test_acc = correct / total

            #Logging metriche su wandb
            if self.logger:
                self.logger.log_metrics({
                    "test/loss": test_loss,
                    "test/accuracy": test_acc,
                })

            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

        return test_losses, test_accuracies

    #Calcola la norma L2 dei gradienti su di un singolo batch
    #per analizzarne le magnitudini
    def gradient_norm(self, dl):
        
        self.model.train()
        inputs, targets = next(iter(dl))
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        logits = self.model(inputs)
        loss = self.criterion(logits, targets)
    
        self.optimizer.zero_grad()
        loss.backward()
            
        self.optimizer.step()

        #Inizializzo due dizionari in cui andrò a salvare 
        #le norme dei gradienti sia per i pesi che per i bias
        grad_weights = {}
        grad_biases = {}
        for name, param in self.model.named_parameters():

            #Utilizzo la norma L2 in quanto è stabile e mi permette di avere
            #un'indicazione chiara sull'intensità del gradiente 
            if param.grad is not None:
                if "weight" in name:
                    grad_weights[name] = param.grad.norm(2).item()
                elif "bias" in name:
                    grad_biases[name] = param.grad.norm(2).item()
    
        sorted_weight_layers = sorted(grad_weights.keys())
        sorted_bias_layers = sorted(grad_biases.keys())
    
        weight_norms = [grad_weights[layer] for layer in sorted_weight_layers]
        bias_norms = [grad_biases[layer] for layer in sorted_bias_layers]
    
        plt.figure(figsize=(12, 5))
        plt.bar(range(len(sorted_weight_layers)), weight_norms, color="blue", label="Weights", alpha=1)
        plt.bar(range(len(sorted_bias_layers)), bias_norms, color="red", label="Biases", alpha=0.8)
    
        plt.xlabel("Layers")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Norm per Layer")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
    
        wandb.log({"Gradient Norm Plot": wandb.Image(plt)})


#Classe che gestisce la distillazione dei modelli come descritto in "Distilling the Knowledge in a Neural Network", NeurIPS 2015
#di Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.
class Distillery_Runner:

    def __init__(self, student, teacher, logger=None, device = "cuda" if torch.cuda.is_available() else "cpu", temperature=3.0, alpha=0.7,):

        #Inizializzo i modelli e li sposto sul device utilizzato
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device
        self.T = temperature
        self.a = alpha
        self.logger = logger
        
        #Funzione di loss sulle hard labels (labels vere)
        self.hard_criterion = nn.CrossEntropyLoss()
        self.soft_criterion = nn.KLDivLoss(reduction='batchmean')
        
        #Ottimizzatore per lo student
        self.optimizer = None

    #Configura l'ottimizzatore per lo student
    def setup_optimizer(self, optimizer_class=optim.Adam, lr=0.001):
            self.optimizer = optimizer_class(self.student.parameters(), lr=lr)

    #Metodo che serve a ottenere i logits del teacher    
    def obtain_teacher_logits(self, dl_train):
            
            teacher_logits = {}
            self.teacher.eval()

            #Non salvo i gradienti in quanto non andrò a modificare il teacher
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(dl_train, desc="Obtaining soft labels from teacher")):
                    inputs = inputs.to(self.device)
                    
                    #Ottieniamo i logits del teacher
                    logits = self.teacher(inputs)
                    
                    #Salviamo i logits usando l'indice del batch
                    teacher_logits[batch_idx] = logits
            
            return teacher_logits

    #Metodo che si occupa della distillazione
    def distillation(self, distill_dl, test_num_epochs):

        self.student.train()

        #Otteniamo i logits dal teacher sul training set
        print("Obtaining logits from teacher...")
        teacher_logits = obtain_teacher_logits(distill_dl)
        
        train_losses = []
        train_accuracies = []

        for epoch in range(test_num_epochs): 
            running_loss, correct, total = 0.0, 0, 0

            for inputs, targets, teacher_logits in tqdm(distill_dl, desc="Distillation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                teacher_logits = teacher_logits.to(self.device)

                #Si ottengono le soft labels pre-computate per il batch corrente
                teacher_logits = teacher_logits[batch_idx].to(self.device)

                #Si ottengono le predizioni dello student 
                student_logits = self.student(inputs)

                #Andiamo a calcolare la soft loss come descritto nel paper 
                soft_targets = F.softmax(teacher_logits / self.T, dim=1)
                soft_probs = F.log_softmax(student_logits / self.T, dim=-1)
                soft_loss = self.soft_criterion(soft_probs, soft_targets) * (self.T * self.T)

                #Si calcola la hard loss
                loss_hard = self.hard_criterion(student_logits, targets)

                #Ciò che avviene è:
                #loss=torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / batch_size
                #Ovvero si confronta quanto le opinioni di student e teacher differiscono
                #e si moltiplica per un valore che quantifica la sicurezza del teacher
                #sulla data previsione.
                #Dunque se il teacher è molto sicuro e lo student sbaglia di molto,
                #la loss avraà un valore considerevole

                #Si pesano le due loss per ottenere la loss finale e,
                #come suggerito nel paper, associamo un peso più basso
                #alle predizioni dello student
                loss = (1 - self.a) * loss_hard + (self.a) * loss_soft

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = student_logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            distill_loss = running_loss / total
            distill_accuracy = correct / total

            distill_losses.append(distill_loss)
            distill_accuracies.append(distill_accuracy)

            if self.logger:
                self.logger.log_metrics({
                    "distill/loss": loss_avg,
                    "distill/accuracy": acc
                }, step=epoch)

        return distill_losses, distill_accuracies
