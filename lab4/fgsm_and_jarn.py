import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from math import log
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import os
import random

#I metodi fsgm_attack, denorm, e fgsm_train sono stati ispirati da https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html
#jarn_train invece fa riferimento invece al paper di Jarn proposto nel notebook, utilizzando tuttavia come scheletro 
#il metodo fgsm come adapter e aggiungendo un discriminatore come proposto nel comando del notebook


#FastGradientSignMethod attacco
def fgsm_attack(img, epsilon, data_grad):
    #Per ogni elemento recupero il segno del gradiente
    sign_data_grad = data_grad.sign()
    #Perturbo l'immagine utilizzando epsilon
    perturbed_image = img + epsilon * sign_data_grad
    #Limitiamo i valori tra 0 e 1
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    #Ritorno l'immagine perturbata
    return perturbed_image

#Ripristina i tensori ai valori originali
def denorm(batch, mean, std, device):
    #Converto media e deviazione standard in tensori
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    #Modello la forma dei tensori affinché possa interagire con [B, C, H, W]
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    #Operazione inversa della normalizzazione (x-mean)/std
    return batch * std + mean


class FGSM_trainer:
    def __init__(self, model, mean, std, logger=None, device=None):
        #Setto il modello
        self.model = model
        #Setto la media e la deviazione standard
        self.mean = mean
        self.std = std
        #Inizializzo il logger
        self.logger = logger

        #Preparo i campi che saranno inizializzati col setup
        self.lr = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.model_type = "classifier"
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    #Metodo che definisce le specifiche dell'esperimento
    def setup_training(self, model_type="classifier", criterion="CrossEntropyLoss", criterion_params=None,
                       optimizer="Adam", lr=0.001, scheduler="CosineAnnealingLR", max_iter=100):

        criterion_params = criterion_params or {}
        self.criterion = getattr(nn, criterion)(**criterion_params)
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr)
        self.scheduler = getattr(lr_scheduler, scheduler)(self.optimizer, max_iter)
        self.lr = lr

    #Addestramento FGSM
    def fgsm_train(self, dl_train, train_epochs_num, epsilon=None):
        losses_clean = []
        losses_adv = []

        self.model.train()

        for epoch in range(train_epochs_num):
            train_bar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{train_epochs_num} [FGSM]", leave=False)
            for data, labels in train_bar:
                data, labels = data.to(self.device), labels.to(self.device)
                data.requires_grad = True

                #Forward dati originali
                if self.model.type == "classifier":
                    outputs_noadv = self.model(data)
                else:
                    _, outputs_noadv = self.model(data)

                noadv_loss = self.criterion(outputs_noadv, labels)

                #Backward sul forward dati originali
                self.optimizer.zero_grad()
                noadv_loss.backward(retain_graph=True)

                data_grad = data.grad.detach()
                 #Denormalizzo per preparare l'attacco
                data_denorm = denorm(data, self.mean, self.std, self.device) 

                #Augmentation con fgsm
                eps = epsilon if epsilon is not None else random.uniform(0.01, 0.3)
                data_adv = fgsm_attack(data_denorm, eps, data_grad)
                #Normalizzo di nuovo
                data_adv = transforms.Normalize(self.mean, self.std)(data_adv)  

                #Forward su dati perturbati
                if self.model.type == "classifier":
                    outputs_adv = self.model(data_adv)
                else:
                    _, outputs_adv = self.model(data_adv)

                loss_adv = self.criterion(outputs_adv, labels)

                #Backward su dati perturbati
                self.optimizer.zero_grad()
                loss_adv.backward()
                self.optimizer.step()

                losses_clean.append(noadv_loss.item())
                losses_adv.append(loss_adv.item())

                if self.logger:
                    self.logger.log_metrics({
                        "FGSM/train_loss_clean": noadv_loss.item(),
                        "FGSM/train_loss_adv": loss_adv.item()
                    }, step=epoch*len(dl_train) + data.size(0))
    
                train_bar.set_postfix(minibatch_noadv_loss=f"{noadv_loss.item():.4f}",
                                  minibatch_loss_adv=f"{loss_adv.item():.4f}")

        return np.mean(losses_clean), np.mean(losses_adv) if len(losses_adv) > 0 else 0

    #Addestramento JARN, utilizzando i parametri specificati nel paper di riferimento per CIFAR10:
    #peso dell'adv loss=1, perturbamento=8/255, learning rate dell'ottimizer del discriminator=2/255
    #e possibilità di utilizzo solo nell'ultimo 25% delle epoche come descritto nel paper
    def jarn_train(self, dl_train, train_epochs_num, discriminator, adv_weight=1, epsilon=8/255, disc_opt_lr=2/255, jarn_start_ratio=0.75):
    
        #Liste per tracciare le loss
        losses_clean = []
        losses_disc = []
    
        #Device
        discriminator.to(self.device)
        self.model.train()
        discriminator.train()
    
        #Calcolo quando iniziare ad applicare JARN
        jarn_start_epoch = int(train_epochs_num * jarn_start_ratio)
    
        #Ottimizzatore discriminatore
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_opt_lr)
        bce = nn.BCEWithLogitsLoss()
    
        for epoch in range(train_epochs_num):
            epoch_loss_clean = []
            epoch_loss_disc = []
    
            train_bar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{train_epochs_num} [JARN]", leave=False)
            apply_jarn = epoch >= jarn_start_epoch 
    
            for step, (data, labels) in enumerate(train_bar):
                data, labels = data.to(self.device), labels.to(self.device)
                data.requires_grad = True
    
                #Forward sul classificatore
                logits_clean = self.model(data)
                loss_cls = self.criterion(logits_clean, labels)
    
                self.optimizer.zero_grad()
                loss_cls.backward(retain_graph=apply_jarn)  #retain_graph solo se serve per JARN
    
                if apply_jarn:
                    #FGSM perturbation
                    data_grad = data.grad.detach()
                    data_denorm = denorm(data, self.mean, self.std, self.device)
                    data_adv = fgsm_attack(data_denorm, epsilon, data_grad)
                    data_adv = transforms.Normalize(self.mean, self.std)(data_adv)
                    data.grad.zero_()
    
                    #Congelo discriminatore
                    for p in discriminator.parameters():
                        p.requires_grad = False
    
                    #Passo al discriminatore (detach per non propagare al classificatore)
                    disc_logits_clean = discriminator(data.detach())
                    disc_logits_adv = discriminator(data_adv.detach())
                    labels_clean = torch.ones_like(disc_logits_clean)
                    labels_adv = torch.zeros_like(disc_logits_adv)
    
                    #Loss discriminatore per regolarizzazione classificatore
                    disc_loss_cls = bce(disc_logits_clean, labels_clean) + bce(disc_logits_adv, labels_adv)
                    total_loss = loss_cls + adv_weight * disc_loss_cls
    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
    
                    #Aggiornamento discriminatore ogni 20 step
                    if step % 20 == 0:
                        for p in discriminator.parameters():
                            p.requires_grad = True
                        disc_optimizer.zero_grad()
                        disc_logits_clean = discriminator(data.detach())
                        disc_logits_adv = discriminator(data_adv.detach())
                        disc_loss_disc = bce(disc_logits_clean, labels_clean) + bce(disc_logits_adv, labels_adv)
                        disc_loss_disc.backward()
                        disc_optimizer.step()
                        epoch_loss_disc.append(disc_loss_disc.item())
                else:
                    #Aggiornamento solo classificatore
                    self.optimizer.step()
                    epoch_loss_disc.append(0.0)
    
                epoch_loss_clean.append(loss_cls.item())
    
            losses_clean.append(np.mean(epoch_loss_clean))
            losses_disc.append(np.mean(epoch_loss_disc))
    
            if self.logger:
                self.logger.log_metrics({
                    "JARN/train_loss_clean": np.mean(epoch_loss_clean),
                    "JARN/train_loss_disc": np.mean(epoch_loss_disc)
                }, step=epoch)
    
        return float(np.mean(losses_clean)), float(np.mean(losses_disc))

    
    #Test per attacco FGSM
    def test_attack(self, test_loader, epsilon):
        self.model.eval()
        adv_examples = []   #Lista per salvare esempi avversari
        total_loss = 0      #Solo per AE
        correct = 0         #Solo per CNN
        y_true = []
        y_pred = []
        ae_scores = []
    
        tqdm_bar = tqdm(test_loader, total=len(test_loader),
                        desc=f"[FGSM attack epsilon:{epsilon}]", leave=False)
        for data, target in tqdm_bar:
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
    
            #Forward originale 
            if self.model.type == "classifier":
                output = self.model(data)
                loss = self.criterion(output, target)
                init_pred = output.max(1, keepdim=True)[1]
                if init_pred.item() != target.item():
                    continue
            else:  #Autoencoder
                _, output = self.model(data)
                loss = self.criterion(output, data).mean()
    
            #Gradiente e perturbazione 
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            data_denorm = denorm(data, self.mean, self.std, self.device)
            perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
            perturbed_data_normalized = transforms.Normalize(self.mean, self.std)(perturbed_data)
    
            #Forward con immagine perturbata 
            if self.model.type == "classifier":
                output = self.model(perturbed_data_normalized)
                loss = self.criterion(output, target).mean()
                final_pred = output.max(1)[1]
    
                #Metriche
                correct += (final_pred == target).sum().item()
                y_true.append(target.view(-1).cpu())
                y_pred.append(final_pred.view(-1).cpu())
    
                #Salvo esempi
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data[0].squeeze().detach().cpu().numpy()
                    adv_examples.append((adv_ex, loss.item()))
    
            else:  #Autoencoder
                _, output = self.model(perturbed_data_normalized)
                loss_adv = self.criterion(output, data).mean()
                total_loss += loss_adv.item()
    
                #Per-sample MSE per OOD
                score = self.criterion(output, data).mean(dim=[1, 2, 3])
                ae_scores.append(-score)
    
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data[0].squeeze().detach().cpu().numpy()
                    adv_examples.append((adv_ex, loss_adv.item()))
    
        #Risultati finali 
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


    #Plotta le metriche e alcuni esempi FGSM
    def plot_result(self, epsilons, examples, metric, model_name, save_path="plot/", n_images=5):
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.plot(epsilons, metric, "*-")
        if self.model.type == "classifier":
            plt.title("Accuracy vs Epsilon Model: " + model_name)
            plt.ylabel("Accuracy")
        else:
            plt.title("Reconstruction Loss vs Epsilon Model: " + model_name)
            plt.ylabel("MSE Loss")
        plt.xlabel("Epsilon")
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'FGSM_eps_{model_name}.png'))
        plt.close()

        plt.figure(figsize=(12, 12))
        plt.suptitle("Model: " + model_name, fontsize=16)
        cnt = 0
        for i, eps in enumerate(epsilons):
            for j in range(min(n_images, len(examples[i]))):
                cnt += 1
                plt.subplot(len(epsilons), n_images, cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel(f"Eps: {eps}", fontsize=14)
                ex, loss_val = examples[i][j]
                plt.title(f"Loss: {loss_val:.3f}")
                img = np.transpose(ex, (1, 2, 0))
                plt.imshow(img)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_path, f'FGSM_EXAMPLE_IMG_{model_name}.png'))
        plt.close()
