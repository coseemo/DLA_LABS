import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics
import os


def plot_confusion_matrix_accuracy(y_gt, y_pred, test_dataloader, model_name, temp, save_path="plots/es1/"):

    #Unisco, rispettivamente, le etichette e le predizioni dei batch in un unico tensore
    y_pred_t = torch.cat(y_pred)
    y_gt_t = torch.cat(y_gt)

    #Calcolo l'accuracy: true/false -> 1/0 -> 1 totali : campioni totali -> media in float
    accuracy = (y_pred_t == y_gt_t).float().mean().item()
    print(f'Accuracy Test Set: {accuracy:.4f}')

    #Calcolo la matrice di confusione
    cm = metrics.confusion_matrix(y_gt_t.cpu(), y_pred_t.cpu())
    #Calcola la percentuale di predizioni corrette per classe
    cmn = (cm.astype(np.float32) / cm.sum(1, keepdims=True)) * 100 
    #Converto in interi per mostrare la heatmap
    cmn = cmn.astype(np.int32)

    #Costruico l'immagine della matrice
    disp = metrics.ConfusionMatrixDisplay(cmn, display_labels=test_dataloader.dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name} - temp {temp} - Accuracy: {accuracy*100:.2f}%")
    plt.savefig(os.path.join(save_path, f"{model_name}_temp:{temp}_confusion_matrix.png"))
    plt.close()


def plot_score(scores_test, scores_fake, model_name, temp, save_path="plot/es1/", score_fun=""):
    
    scores_test = scores_test.flatten()
    scores_fake = scores_fake.flatten()

    #Line plot dei punteggi ordinati
    plt.figure()
    plt.plot(sorted(scores_test), label='test')
    plt.plot(sorted(scores_fake), label='fake')
    plt.legend()
    plt.title(f"Score - {model_name} - temp {temp} - {score_fun}")
    plt.savefig(os.path.join(save_path, f"{model_name}_temp:{temp}__score_{score_fun}.png"))
    plt.close()

    #Istogrammi dei punteggi test vs fake
    plt.figure()
    plt.hist(scores_test, bins=25, density=True, alpha=0.5, label='test')
    plt.hist(scores_fake, bins=25, density=True, alpha=0.5, label='fake')
    plt.legend()
    plt.title(f"Histogram - {model_name} - temp {temp} - {score_fun}")
    plt.savefig(os.path.join(save_path, f"{model_name}_temp {temp}_score_hist_{score_fun}.png"))
    plt.close()

    #Unisco le predizioni e creo delle etichette 1 per le vere e 0 per le false
    y_pred = torch.cat((scores_test, scores_fake))
    y_true = torch.cat((torch.ones_like(scores_test), torch.zeros_like(scores_fake)))

    #Curva ROC
    fig = metrics.RocCurveDisplay.from_predictions(y_true, y_pred).figure_
    fig.axes[0].set_title(f"ROC Curve - {model_name} - temp {temp} - {score_fun}")
    fig.savefig(os.path.join(save_path, f"{model_name}_temp:{temp}__roc_curve_{score_fun}.png"))
    plt.close(fig)

    #Curva Precision-Recall
    fig = metrics.PrecisionRecallDisplay.from_predictions(y_true, y_pred).figure_
    fig.axes[0].set_title(f"Precision-Recall Curve - {model_name} - temp {temp} - {score_fun}")
    fig.savefig(os.path.join(save_path, f"{model_name}_temp:{temp}__precision_recall_curve_{score_fun}.png"))
    plt.close(fig)


def plot_logit_softmax(x, k, model, device, model_name, temp, save_path="plot/es1/", ty=""):

    #Calcolo i logits del modello
    output = model(x.to(device))
    #Estraggo i logits di k
    logits = output[k].detach().cpu()
    #Esatraggo il valore softmax di k
    softmax_vals = F.softmax(output, dim=1)[k].detach().cpu()

    #Logits
    plt.figure()
    plt.bar(np.arange(len(logits)), logits)
    plt.title(f'Logit - {model_name} - temp {temp} - {ty}')
    plt.savefig(os.path.join(save_path, f"{model_name}_temp:{temp}__logit_{ty}.png"))
    plt.close()

    #Softmax
    plt.figure()
    plt.bar(np.arange(len(softmax_vals)), softmax_vals)
    plt.title(f'Softmax - {model_name} - temp {temp} - {ty}')
    plt.savefig(os.path.join(save_path, f"{model_name}_temp:{temp}__softmax_{ty}.png"))
    plt.close()

    #Input image
    plt.figure()
    #Si permuta per matplotlib
    img = x[k].permute(1, 2, 0).cpu()
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Input Image - {ty}")
    plt.savefig(os.path.join(save_path, f"{model_name}_temp:{temp}__input_{ty}.png"))
    plt.close()

def compute_scores(model, dataloader, score_fun, device):
    scores = []
    with torch.no_grad():
        tqdm_bar = tqdm(dataloader, desc="[Testing (Val/Test/Fake)]", leave=False)
        for data in tqdm_bar:
            x, y = data
            output = model(x.to(device))
            s = score_fun(output)
            scores.append(s)
        scores_t = torch.cat(scores)
        return scores_t

def max_logit(logit):
    #Ottiene il massimo per ogni elemento del batch
    s = logit.max(dim=1)[0] 
    return s


def max_softmax(logit, t = 1.0):
    s = F.softmax(logit/t, 1)
    #get the max for each element of the batch
    s = s.max(dim=1)[0] 
    return s
