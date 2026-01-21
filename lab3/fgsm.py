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
from sklearn.metrics import accuracy_score

#I metodi fsgm_attack e denorm sono stati ispirati da https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html

#Attacco FastGradientSignMethod 
def fgsm_attack(img, epsilon, data_grad):
    #Per ogni elemento recupero il segno del gradiente
    sign_data_grad = data_grad.sign()
    #Perturbo l'immagine utilizzando epsilon 
    #(+eps massimizza distanza dall'originale, 
    #-eps minimizza distanza da target attack)
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
        #Device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        #Preparo i campi che saranno inizializzati col setup
        self.fgsm = False
        self.epsilon = None
        self.lr = None
        self.optimizer = None
        self.scheduler = None
        
        #Loss basati sul tipo di modello
        if model.type == "autoencoder":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    #Metodo che definisce le specifiche dell'esperimento
    def setup_training(self, optimizer="Adam", fgsm=False, epsilon=None, scheduler="CosineAnnealingLR", 
                       lr=0.0001, max_iter=200):

        self.lr = lr
        self.fgsm = fgsm
        self.epsilon = epsilon
        
        #Setup dell'optimizer
        optimizer_class = getattr(optim, optimizer)
        self.optimizer = optimizer_class(self.model.parameters(), lr)
        
        #Setup dello scheduler
        if scheduler:
            scheduler_class = getattr(lr_scheduler, scheduler)
            self.scheduler = scheduler_class(self.optimizer, max_iter)
        else:
            self.scheduler = None

    def setup_attack(self, epsilon):
        self.epsilon = epsilon
        
    #Metodo per il training delle CNN
    def train_classifier(self, dataloader, epoch=0):
        
        self.model.train()
        model = self.model.to(self.device)
    
        losses_clean = []
        losses_adv = []
    
        train_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Training]", leave=False)
        for data, labels in train_bar:
            data, labels = data.to(self.device), labels.to(self.device)

            #Se l'allenamento avversariale è attivo, tengo traccia dei gradienti
            if self.fgsm:
                data.requires_grad = True
    
            logits_clean = model(data)
            loss_clean = self.criterion(logits_clean, labels)

            #Allenamento avversariale: train su immagini pulite + immagini perturbate
            if self.fgsm:
    
                #Backward sul forward clean
                model.zero_grad()
                loss_clean.backward()
                self.optimizer.step()

                #Salvo i gradienti e denormalizzo per preparare l'attacco
                data_grad = data.grad.data
                data_denorm = denorm(data, self.mean, self.std, self.device)

                #Se nelle config epsilon = null, scelgo un epsilon randomico tra 0.01 e 0.15 per ogni batch
                if self.epsilon is None:
                    epsilon = random.uniform(0.01, 1.5)
                    data_adv = fgsm_attack(data_denorm, epsilon, data_grad) #Faccio augmentation con fgsm e epsilon random
                else:
                    data_adv = fgsm_attack(data_denorm, self.epsilon, data_grad) #Faccio augmentation con fgsm e epsilon statico

                #Ri-normalizzo i dati adv
                data_adv = transforms.Normalize(self.mean, self.std)(data_adv)
    
                #Forward sugli adv
                logits_adv = model(data_adv)
                loss_adv = self.criterion(logits_adv, labels)

                self.optimizer.zero_grad()
                loss_adv.backward()
                self.optimizer.step()
    
                losses_clean.append(loss_clean.item())
                losses_adv.append(loss_adv.item())
    
                train_bar.set_postfix(minibatch_loss_clean=f"{loss_clean.item():.4f}", minibatch_loss_adv=f"{loss_adv.item():.4f}")

            #Training normale
            else:

                self.optimizer.zero_grad()
                loss_clean.backward()
                self.optimizer.step()
                
                losses_clean.append(loss_clean.item())
                train_bar.set_postfix(minibatch_loss_clean=f"{loss_clean.item():.4f}")

        #Ritorno loss media su dati clean e adv (0 se non ho fatto adversarial training)
        return np.mean(losses_clean), np.mean(losses_adv) if len(losses_adv) > 0 else 0

    #Metodo per il training degli autoencoders
    def train_autoencoder(self, dataloader, epoch=0):
        
        self.model.train()
        model = self.model.to(self.device)
    
        losses_clean = []
        losses_adv = []
    
        train_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Training]", leave=False)
        for data, _ in train_bar:
            data = data.to(self.device)
    
            if self.fgsm:
                data.requires_grad = True
    
            self.optimizer.zero_grad()

            #Forward: encoding + ricostruzione
            z, x_rec = model(data)
            loss_clean = self.criterion(data, x_rec)  #MSE tra input e ricostruzione
    
            if self.fgsm:
    
                #Backward per ottenere gradienti rispetto all'input
                model.zero_grad()
                loss_clean.backward()
                self.optimizer.step()

                #Salvo i gradienti e denormalizzo per preparare l'attacco
                data_grad = data.grad.data
                data_denorm = denorm(data, self.mean, self.std, self.device) 

                #Se nelle config epsilon = null, scelgo un epsilon randomico tra 0.01 e 0.15 per ogni batch
                if self.epsilon is None:
                    epsilon = random.uniform(0.01, 0.15)
                    data_adv = fgsm_attack(data_denorm, epsilon, data_grad) 
                else:
                    data_adv = fgsm_attack(data_denorm, self.epsilon, data_grad) 
    
                data_adv = transforms.Normalize(self.mean, self.std)(data_adv) 
    
                #Forward sugli esempi adv
                z_adv, x_rec_adv = model(data_adv)
                loss_adv =  self.criterion(data, x_rec_adv) #MSE tra input originale e ricostruzione dell'adversarial
    
                #Backward sul forward adv
                self.optimizer.zero_grad()
                loss_adv.backward()
                self.optimizer.step()
    
                losses_clean.append(loss_clean.item())
                losses_adv.append(loss_adv.item())
    
                train_bar.set_postfix(minibatch_loss_clean=f"{loss_clean.item():.4f}", minibatch_loss_adv=f"{loss_adv.item():.4f}")

            #Allenamento standard
            else:
    
                self.optimizer.zero_grad()
                loss_clean.backward()
                self.optimizer.step()
                losses_clean.append(loss_clean.item())
                train_bar.set_postfix(minibatch_loss_clean=f"{loss_clean.item():.4f}")
    
    
        return np.mean(losses_clean), np.mean(losses_adv) if len(losses_adv) > 0 else 0
    
    #Metodo per il test delle CNN
    def test_classifier(self, dataloader):
        self.model.eval()
        predictions = []
        gts = []
        losses = []
    
        test_bar = tqdm(dataloader, desc="[Test/Validation]", leave=False)
        with torch.no_grad():
            for data, labels in test_bar:
                data = data.to(self.device)
                labels = labels.to(self.device)
    
                logits = self.model(data)
                loss = self.criterion(logits, labels)
                #Converto logits in probabilità e prendo la classe con prob massima
                prob = F.softmax(logits, dim=1)
                pred = torch.argmax(prob, dim=1)

                #Raccolgo ground truth e predizioni per calcolare accuracy
                gts.append(labels.cpu().numpy())
                predictions.append(pred.cpu().numpy())
                losses.append(loss.item())
                test_bar.set_postfix(minibatch_loss=f"{loss.item():.4f}")

        #Accuracy complessiva su tutto il test set
        final_accuracy = accuracy_score(np.hstack(gts), np.hstack(predictions))
        avg_loss = np.mean(losses)
    
        return final_accuracy, avg_loss
    
    #Metodo per prendere le predizioni delle CNN 
    #(serve per plottare la matrice di confusione)
    def get_pred_classifier(self, dataloader):
        self.model.eval()
        y_gt, y_pred = [], []
        
        for data, labels in dataloader:
            x, y = data.to(self.device), labels.to(self.device)
    
            with torch.no_grad():
                yp = self.model(x)

            #Salvo classe predetta (argmax) e ground truth
            y_pred.append(yp.argmax(1))
            y_gt.append(y)
    
        return y_gt, y_pred
    
    
    #Metodo di test per gli Autoencoder
    def test_autoencoder(self, dataloader):
        self.model.eval()
        #MSE pixel-wise (no media)
        loss = nn.MSELoss(reduction='none')
        scores = []
        losses = []
        
        tqdm_bar = tqdm(dataloader, desc="[Testing (Val/Test/Fake)]", leave=False)
        with torch.no_grad():
            for data, _ in tqdm_bar:
                x = data.to(self.device)
                #Encoding + ricostruzione
                z, xr = self.model(x)
                #MSE pixel-wise tra input e ricostruzione
                l = loss(x, xr)
                #Score di anomalia: media MSE su ogni immagine (alto MSE = anomalia)
                score = l.mean([1, 2, 3])
    
                losses.append(score)
                scores.append(-score) #Negativo perché score alto = anomalia (per threshold)

    
        scores = torch.cat(scores)
        losses = torch.mean(torch.cat(losses))
        
        return scores, losses.item()

    #Metodo per testare un attco FGSM sulle CNN    
    def test_attack_classifier(self, test_loader):
        self.model.eval()
        correct = 0
        adv_examples = []
        total = 0
        
        tqdm_bar = tqdm(test_loader, total=len(test_loader), desc=f"[FGSM attack epsilon: {self.epsilon}]", leave=False)
        for data, target in tqdm_bar:
            data, target = data.to(self.device), target.to(self.device)
            total += 1
    
            data.requires_grad = True
    
            #Forward sul modello
            output = self.model(data)
            #Indice della max log-probability
            init_pred = output.max(1, keepdim=True)[1]
            #Calcolo la loss
            loss = self.criterion(output, target)
            #Setto a zero i gradienti del modello
            self.model.zero_grad()
            #Calcolo i gradienti del modello in backward pass
            loss.backward()
            #Li salvo
            data_grad = data.grad.data
            #Riporto i dati alla loro scala originale
            data_denorm = denorm(data, self.mean, self.std, self.device)
            #Perturbo il dato con fgsm
            perturbed_data = fgsm_attack(data_denorm, self.epsilon, data_grad)
            #Applico di nuovo la normalizzaione
            perturbed_data_normalized = transforms.Normalize(self.mean, self.std)(perturbed_data)
    
            #Classifico l'immagine perturbata
            with torch.no_grad():
                output = self.model(perturbed_data_normalized)
    
            #Guardo se la predizione è corretta
            final_pred = output.max(1, keepdim=True)[1] 
            if final_pred.item() == target.item():
                correct += 1
    
            #Salvo alcuni esempi
            if len(adv_examples) < 5:
                orig = data_denorm.squeeze().detach().cpu().numpy()
                adv = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv, orig))
        
        final_acc = correct / float(total) if total > 0 else 0.0
        return final_acc, adv_examples


    def test_attack_autoencoder(self, test_loader):
        self.model.eval()
        total_loss = 0
        adv_examples = []
    
        for data, _ in tqdm(test_loader, leave=False):
            data = data.to(self.device)
            data.requires_grad = True
    
            #Forward
            z, x_rec = self.model(data)
            loss = self.criterion(x_rec, data)
    
            #Backward
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
    
            #Attaco con fgsm
            data_denorm = denorm(data, self.mean, self.std, self.device)
            perturbed_data = fgsm_attack(data_denorm, self.epsilon, data_grad)
            perturbed_data_normalized = transforms.Normalize(self.mean, self.std)(perturbed_data)
    
            #Forward pass su dati perturbati
            z_adv, x_rec_adv = self.model(perturbed_data_normalized)
            adv_loss = self.criterion(x_rec_adv, data)
            total_loss += adv_loss.item()
    
            #Salva qualche esempio per il plot
            if len(adv_examples) < 5:
                orig = data_denorm.squeeze().detach().cpu().numpy()
                adv = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((loss.item(), adv_loss.item(), adv, orig))

        avg_loss = total_loss / len(test_loader)

        return avg_loss, adv_examples
    
    #Attacco mirato fgsm
    def targeted_attack_classifier(self, test_loader, target_class):
        
        self.model.eval()
        total_samples = 0
        correctly_classified_original = 0
        targeted_success = 0
        targeted_success_from_correct = 0
        adv_examples = []
        
        tqdm_bar = tqdm(
            test_loader, 
            total=len(test_loader),
            desc=f"[Targeted FGSM attack epsilon: {self.epsilon} -> target:{target_class}]",
            leave=False
        )
    
        for data, target in tqdm_bar:
            data, target = data.to(self.device), target.to(self.device)
    
            #Skip se il modello predice già target_class
            with torch.no_grad():
                output = self.model(data)
                pred = output.max(1)[1]
                
                if (pred == target_class).any():
                    continue

            #Traccia se era classificato correttamente
            was_correct = (pred.item() == target.item())
        
            data.requires_grad = True
    
            #Sostituisco la label con target_class
            target_labels = torch.full_like(target, target_class)
    
            #Forward
            output = self.model(data)
    
            #Targeted loss
            loss = self.criterion(output, target_labels)
    
            self.model.zero_grad()
            loss.backward()
    
            data_grad = data.grad.data
    
            #Solita de-normalizzazione
            data_denorm = denorm(data, self.mean, self.std, self.device)
            #FGSM mirato: -epsilon minimizza la loss verso target_class (gradiente opposto per attrarre verso il target)
            perturbed_data = fgsm_attack(data_denorm, -self.epsilon, data_grad)
            #Ri-normalizzo
            perturbed_data_norm = transforms.Normalize(self.mean, self.std)(perturbed_data)
    
            #Classifico l’immagine avversaria
            with torch.no_grad():
                output_adv = self.model(perturbed_data_norm)
                final_pred = output_adv.max(1)[1]
                confidence = torch.softmax(output_adv, dim=1)[0, target_class].item()
    
            #Aggiorna metriche
            total_samples += 1
            if was_correct:
                correctly_classified_original += 1
            
            attack_success = (final_pred.item() == target_class)
            if attack_success:
                targeted_success += 1
                if was_correct:
                    targeted_success_from_correct += 1
    
            #Salva esempi (sia successi che fallimenti per analisi)
            if len(adv_examples) < 10:
                orig = data_denorm.squeeze().detach().cpu().numpy()
                adv = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append({
                    'original_class': pred.item(),
                    'true_class': target.item(),
                    'adversarial_class': final_pred.item(),
                    'target_class': target_class,
                    'success': attack_success,
                    'confidence': confidence,
                    'original_image': orig,
                    'adversarial_image': adv,
                    'perturbation': adv - orig
                })
    
        #Risultati dettagliati
        results = {
            #Metrica che dice: percentuale di successo sui campioni che non erano già target_class (che il modello li classificasse bene o meno)
            'overall_success_rate': targeted_success / total_samples if total_samples > 0 else 0.0,
            #Metrica che dice: percentuale di successo sui campioni che non erano già target_class e che il modello classificava correttamente
            'success_from_correct': targeted_success_from_correct / correctly_classified_original if correctly_classified_original > 0 else 0.0,
            #Metrica che dice: numero totale di campioni processati (esclusi gli skip)
            'total_samples': total_samples,
            #Metrica che dice: quanti esempi erano classificati correttamente tra quelli processati
            'correctly_classified': correctly_classified_original,
            #Metrica che dice: quanti attacchi sono riusciti (tra i campioni processati)
            'targeted_successes': targeted_success
        }
        
        return results, adv_examples

    #Plotta le metriche e alcuni esempi FGSM
    def plot_result(self, epsilons, examples, metric, model_name, save_path="plot/", n_images=5):
        
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(5, 5))
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
        plt.suptitle("Model: " + model_name)
        cnt = 0
        for i in range(len(epsilons)):
            for j in range(len(examples[i])):
                cnt += 1
                plt.subplot(len(epsilons), n_images, cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                
                if j == 0:
                    plt.ylabel(f"Eps: {epsilons[i]}")

                if self.model.type == "classifier":
                    orig_pred, adv_pred, adv_img, _ = examples[i][j]
                    plt.title(f"{orig_pred} -> {adv_pred}")
                    img = np.transpose(adv_img, (1, 2, 0))
                    
                else:
                    orig_loss, adv_loss, adv_img, _ = examples[i][j]
                    plt.title(f"MSE: {orig_loss:.3f} -> {adv_loss:.3f}")
                    img = np.transpose(adv_img, (1, 2, 0))

                plt.imshow(img)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_path, f'FGSM_EXAMPLE_IMG_{model_name}.png'))
        plt.close()

    #Questi ultimi tre metodi per il plot dei risultati sono stati scritti interagendo anche con Claude (sonnet 4.5)
    def plot_target_attack(self, epsilons, attack_success_rate, target_class, model_name, save_path="plot/"):
    
        plt.figure(figsize=(8, 5))
        plt.plot(epsilons, attack_success_rate, marker="o")
        plt.xlabel("Epsilon")
        plt.ylabel("Attack Success Rate")
        plt.title("Attack Success Rate per Epsilon - Target class:" + str(target_class) + "-Model:" + model_name)
        plt.ylim([0, 1])
        plt.savefig(save_path + 'FGSM_SUCCESS_RATE_TARGET_'+str(target_class)+'_' + '.png')
 
    def plot_targeted_examples(self, adv_examples, epsilon, model_name, save_path="plot/"):
        
        # Separa successi e fallimenti
        successes = [ex for ex in adv_examples if ex['success']]
        failures = [ex for ex in adv_examples if not ex['success']]
        
        n_success = min(3, len(successes))
        n_fail = min(2, len(failures))
        
        fig, axes = plt.subplots(n_success + n_fail, 3, figsize=(16, 4*(n_success + n_fail)))
        if n_success + n_fail == 1:
            axes = axes.reshape(1, -1)
        
        row = 0
        
        # Visualizza successi
        for i, ex in enumerate(successes[:n_success]):
            #Originale
            axes[row, 0].imshow(ex['original_image'].transpose(1, 2, 0))
            axes[row, 0].set_title(f"Original\nPred: {ex['original_class']}\nTrue: {ex['true_class']}", 
                                   fontsize=10, color='blue')
            axes[row, 0].axis('off')
            
            #Perturbazione (normalizzata per visibilità)
            pert = ex['perturbation'].transpose(1, 2, 0)
            pert_norm = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
            axes[row, 1].imshow(pert_norm)
            axes[row, 1].set_title(f"Perturbation\nε={epsilon}", fontsize=10)
            axes[row, 1].axis('off')
            
            #Immagine Perturbata
            axes[row, 2].imshow(ex['adversarial_image'].transpose(1, 2, 0))
            axes[row, 2].set_title(f"Adversarial\nPred: {ex['adversarial_class']} ✓", 
                                   fontsize=10, color='green')
            axes[row, 2].axis('off')
            
            row += 1
        
        #Visualizza fallimenti (se presenti)
        for i, ex in enumerate(failures[:n_fail]):
            axes[row, 0].imshow(ex['original_image'].transpose(1, 2, 0))
            axes[row, 0].set_title(f"Original\nPred: {ex['original_class']}", fontsize=10)
            axes[row, 0].axis('off')
            
            pert = ex['perturbation'].transpose(1, 2, 0)
            pert_norm = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
            axes[row, 1].imshow(pert_norm)
            axes[row, 1].set_title(f"Perturbation\nε={epsilon}", fontsize=10)
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(ex['adversarial_image'].transpose(1, 2, 0))
            axes[row, 2].set_title(f"FAILED\nPred: {ex['adversarial_class']} ✗", 
                                   fontsize=10, color='red')
            axes[row, 2].axis('off')
            
            row += 1
        
        target_class = successes[0]['target_class'] if successes else failures[0]['target_class']
        plt.suptitle(f"Targeted Attack -> Class {target_class} | Model: {model_name} | eps={epsilon}", 
                     fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(f"{save_path}targeted_examples_class{target_class}_{model_name}_eps{epsilon}.png", 
                    bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Salvati {n_success} successi e {n_fail} fallimenti")

    def print_attack_summary(self, results, target_class, epsilon):
        
        print(f"\n{'='*60}")
        print(f"TARGETED ATTACK SUMMARY - Target Class: {target_class} | ε={epsilon}")
        print(f"{'='*60}")
        print(f"Total samples tested: {results['total_samples']}")
        print(f"Originally correct:   {results['correctly_classified']} ({results['correctly_classified']/results['total_samples']*100:.1f}%)")
        print(f"Attack successes:     {results['targeted_successes']} ({results['overall_success_rate']*100:.1f}%)")
        print(f"Success from correct: {results['success_from_correct']*100:.1f}%")
        print(f"{'='*60}\n")