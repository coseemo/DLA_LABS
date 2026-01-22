# Laboratory 4: OOD, FGSM, Targeted FGSM

## Organization

The core implementation is primarily contained in fgsm.py. Experimental configuration and execution are managed in main.py, and all metrics are implemented in metrics.py.

## Plots

All the plots can be found here: [<https://wandb.ai/cosimo-borghini1-universit-di-firenze/LAB4-OOD_Detection?nw=nwusercosimoborghini1>]

## Pretraining

### Parameters

To run this experiment use:

```         
python main.py --experiment pretrain
```

For these experiment I use this configuration, that can be found in
`/configs/config_pretrain.yaml`:

```         
#Configurazioni globali
global_configs:

    seed: 99
    device: auto
    
    batch_size: 256
    validation_split: 10
    num_workers: 2
    mean: [0.4914, 0.4822, 0.4465]
    std: [0.2470, 0.2435, 0.2616]
    
    project_name: "Lab4-OOD_Detection"

#Configurazione per il pretraining
pretraining_configs:

  #true/false permette di testare solo alcuni
  #dei modelli elencati
  
  pretrain_cnn_plus: true
  pretrain_autoencoder: true
  pretrain_cnn: true
  
  #Lista dei modelli CNN 
  cnn_models:
  
  #NOTA: 
  #fgsm gestisce l'allenamento con o senza esempi adv
  #quando epsilon = null significa che epsilon è 
  #un valore random tra (0.01-0.15)
  
    - name: "CNN"
      path: "models/CNN.pth"
      fgsm: false
      epochs: 200
      epsilon: 0
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
      
    - name: "CNN_0.05"
      path: "models/CNN_0.05.pth"
      fgsm: true
      epochs: 200
      epsilon: 0.05
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
      
    - name: "CNN_0.1"
      path: "models/CNN_0.1.pth"
      fgsm: true
      epochs: 200
      epsilon: 0.1
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
      
    - name: "CNN_None"
      path: "models/CNN_None.pth"
      fgsm: true
      epochs: 200
      epsilon: null
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"

  #Lista modelli CNNplus 
  cnn_plus_models:  
  
    - name: "CNNplus"
      path: "models/CNNplus.pth"
      fgsm: false
      epochs: 50
      epsilon: 0
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
      
    - name: "CNNplus_0.05"
      path: "models/CNNplus_0.05.pth"
      fgsm: true
      epochs: 50
      epsilon: 0.05
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
      
    - name: "CNNplus_0.1"
      path: "models/CNNplus_0.1.pth"
      fgsm: true
      epochs: 50
      epsilon: 0.1
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
      
    - name: "CNNplus_None"
      path: "models/CNNplus_None.pth"
      fgsm: true
      epochs: 50
      epsilon: null
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
  
  #Lista dei modelli Autoencoder 
  autoencoder_models:
  
    - name: "Autoencoder"
      path: "models/Autoencoder.pth"
      fgsm: false
      epochs: 200
      epsilon: 0
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
      
    - name: "Autoencoder_0.05"
      path: "models/Autoencoder_0.05.pth"
      fgsm: true
      epochs: 200
      epsilon: 0.05
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
      
    - name: "Autoencoder_0.1"
      path: "models/Autoencoder_0.1.pth"
      fgsm: true
      epochs: 200
      epsilon: 0.1
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
      
    - name: "Autoencoder_None"
      path: "models/Autoencoder_None.pth"
      fgsm: true
      epochs: 200
      epsilon: null
      optimizer: "Adam"
      lr: 0.0001
      scheduler: "CosineAnnealingLR"
```

### Results of the Training

All models (CNN, CNNplus, and Autoencoder) were evaluated both with and
without adversarial training. When adversarial training was applied,
several epsilon values were considered 0.1,0.05,random(0.01–0.15). It is worth noting
that eps = null corresponds to a randomly sampled epsilon in the range
(0.1–0.15) for each batch.

-   **CNN**

    -   Simpler architecture

    -   Converges in \~200 epochs

    -   Uses Adam optimizer (lr=0.0001) with cosine annealing scheduler

    -   Achieves reasonable accuracy, but lower than CNNplus

| Train Loss Adv                                               | Train Loss Clean                                                 |
| ------------------------------------------------------------ | ---------------------------------------------------------------- |
| ![CNN Train Loss Adv](plots/pretrain/cnn_train_loss_adv.png) | ![CNN Train Loss Clean](plots/pretrain/cnn_train_loss_clean.png) |

| Val Loss                                         | Test Accuracy                                              |
| ------------------------------------------------ | ---------------------------------------------------------- |
| ![CNN Val Loss](plots/pretrain/cnn_val_loss.png) | ![CNN Test Accuracy](plots/pretrain/cnn_test_accuracy.png) |


      
-   **CNNplus**

    -   More expressive and stable architecture

    -   Converges faster (\~50 epochs)

    -   Uses Adam optimizer (lr=0.0001) with cosine annealing scheduler

    -   Achieves higher accuracy than CNN

| Train Loss Adv                                                         | Train Loss Clean                                                           |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| ![CNN Plus Train Loss Adv](plots/pretrain/cnn_plus_train_loss_adv.png) | ![CNN Plus Train Loss Clean](plots/pretrain/cnn_plus_train_loss_clean.png) |

| Test Accuracy                                                   | Val Loss                                                   |
| --------------------------------------------------------------- | ---------------------------------------------------------- |
| ![CNN Plus Test Accuracy](plots/pretrain/cnn_plus_test_acc.png) | ![CNN Plus Val Loss](plots/pretrain/cnn_plus_val_loss.png) |


-   **AutoEncoder**

    -   Trained for 200 epochs

    -   Uses Adam optimizer (lr=0.0001) with cosine annealing scheduler

    -   Loss function: Mean Squared Error (MSELoss) on reconstruction
        output
        
| Train Loss Adv                                                               | Train Loss Clean                                                                 |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| ![Autoencoder Train Loss Adv](plots/pretrain/autoencoder_train_loss_adv.png) | ![Autoencoder Train Loss Clean](plots/pretrain/autoencoder_train_loss_clean.png) |

| Val Loss                                                         |   |
| ---------------------------------------------------------------- | - |
| ![Autoencoder Val Loss](plots/pretrain/autoencoder_val_loss.png) |   |



## Experiment 1e2

### Parameters

To run this experiment use:

```         
python main.py --experiment 1e2
```

For these experiment I use this configuration, that can be found in
`/configs/config_1.yaml`:

```         
seed: 99
device: auto

data:
  batch_size: 256
  validation_split: 10
  num_workers: 2
  mean: [0.4914, 0.4822, 0.4465]
  std: [0.2023, 0.1994, 0.2010]

#Configurazione per l'esperimento 1 e 2
models:
  test_cnn: true
  test_autoencoder: true
  
  #Lista dei modelli CNN da testare
  cnn_models:
    - name: "CNN"
      path: "models/CNN.pth"
    - name: "CNNplus"
      path: "models/CNNplus.pth"
    - name: "CNN_0.05"
      path: "models/CNN_0.05.pth"
    - name: "CNNplus_0.05"
      path: "models/CNNplus_0.05.pth"
    - name: "CNN_0.1"
      path: "models/CNN_0.1.pth"
    - name: "CNNplus_0.1"
      path: "models/CNNplus_0.1.pth"
    - name: "CNN_None"
      path: "models/CNN_None.pth"
    - name: "CNNplus_None"
      path: "models/CNNplus_None.pth"
  
  #Lista dei modelli Autoencoder da testare
  autoencoder:
    - name: "Autoencoder"
      path: "models/Autoencoder.pth"
    - name: "Autoencoder_0.05"
      path: "models/Autoencoder_0.05.pth"
    - name: "Autoencoder_0.1"
      path: "models/Autoencoder_0.1.pth"
    - name: "Autoencoder_None"
      path: "models/Autoencoder_None.pth"

#Parametri per la valutazione
evaluation:
  temperature: 1000

fgsm:
  epsilons_cnn: [0.0, 0.05, 0.075, 0.1, 0.125, 0.15]
  epsilons_ae: [0.0, 0.05, 0.075, 0.1, 0.125, 0.15]

logging:
  project_name: "Lab4-OOD_Detection"
```

The commented parameters are the ones used for the various runs.

### Results Exercise 1

#### CNN

<details>

<summary><strong>📊 CNN with no FGSM training</strong></summary>

<br>

|  |  |
|---------|----------|
| Confusion Matrix | ![Confusion Matrix](plots/es1e2/CNN/CNN_temp:1000_confusion_matrix.png) |
| Input Fake Data | ![Input Fake](plots/es1e2/CNN/CNN_temp:1000__input_Fake_data.png) |
| Input Real Data | ![Input Real](plots/es1e2/CNN/CNN_temp:1000__input_Real_data.png) |
| Logit Fake Data | ![Logit Fake](plots/es1e2/CNN/CNN_temp:1000__logit_Fake_data.png) |
| Logit Real Data | ![Logit Real](plots/es1e2/CNN/CNN_temp:1000__logit_Real_data.png) |
| Precision-Recall (Max Logit) | ![PR Logit](plots/es1e2/CNN/CNN_temp:1000__precision_recall_curve_max_logit.png) |
| Precision-Recall (Max Softmax) | ![PR Softmax](plots/es1e2/CNN/CNN_temp:1000__precision_recall_curve_max_softmax.png) |
| ROC Curve (Max Logit) | ![ROC Logit](plots/es1e2/CNN/CNN_temp:1000__roc_curve_max_logit.png) |
| ROC Curve (Max Softmax) | ![ROC Softmax](plots/es1e2/CNN/CNN_temp:1000__roc_curve_max_softmax.png) |
| Score Histogram (Max Logit) | ![Score Hist Logit](plots/es1e2/CNN/CNN_temp:1000_score_hist_max_logit.png) |
| Score Histogram (Max Softmax) | ![Score Hist Softmax](plots/es1e2/CNN/CNN_temp:1000_score_hist_max_softmax.png) |
| Score (Max Logit) | ![Score Max Logit](plots/es1e2/CNN/CNN_temp:1000__score_max_logit.png) |
| Score (Max Softmax) | ![Score Max Softmax](plots/es1e2/CNN/CNN_temp:1000__score_max_softmax.png) |
| Softmax Fake Data | ![Softmax Fake](plots/es1e2/CNN/CNN_temp:1000__softmax_Fake_data.png) |
| Softmax Real Data | ![Softmax Real](plots/es1e2/CNN/CNN_temp:1000__softmax_Real_data.png) |
</details>

<details>

<summary><strong>📊 CNN with eps=0.05 FGSM
training</strong></summary>

<br>

|  |  |
|---------|----------|
| Confusion Matrix | ![Confusion Matrix](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000_confusion_matrix.png) |
| Input Fake Data | ![Input Fake](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__input_Fake_data.png) |
| Input Real Data | ![Input Real](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__input_Real_data.png) |
| Logit Fake Data | ![Logit Fake](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__logit_Fake_data.png) |
| Logit Real Data | ![Logit Real](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__logit_Real_data.png) |
| Precision-Recall (Max Logit) | ![PR Logit](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__precision_recall_curve_max_logit.png) |
| Precision-Recall (Max Softmax) | ![PR Softmax](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__precision_recall_curve_max_softmax.png) |
| ROC Curve (Max Logit) | ![ROC Logit](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__roc_curve_max_logit.png) |
| ROC Curve (Max Softmax) | ![ROC Softmax](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__roc_curve_max_softmax.png) |
| Score Histogram (Max Logit) | ![Score Hist Logit](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000_score_hist_max_logit.png) |
| Score Histogram (Max Softmax) | ![Score Hist Softmax](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000_score_hist_max_softmax.png) |
| Score (Max Logit) | ![Score Max Logit](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__score_max_logit.png) |
| Score (Max Softmax) | ![Score Max Softmax](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__score_max_softmax.png) |
| Softmax Fake Data | ![Softmax Fake](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__softmax_Fake_data.png) |
| Softmax Real Data | ![Softmax Real](plots/es1e2/CNN_0.05/CNN_0.05_temp:1000__softmax_Real_data.png) |

</details>

</details>

<details>

<summary><strong>📊 CNN with eps=0.1 FGSM
training</strong></summary>

<br>

|  |  |
|---------|----------|
| Confusion Matrix | ![Confusion Matrix](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000_confusion_matrix.png) |
| Input Fake Data | ![Input Fake](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__input_Fake_data.png) |
| Input Real Data | ![Input Real](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__input_Real_data.png) |
| Logit Fake Data | ![Logit Fake](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__logit_Fake_data.png) |
| Logit Real Data | ![Logit Real](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__logit_Real_data.png) |
| Precision-Recall (Max Logit) | ![PR Logit](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__precision_recall_curve_max_logit.png) |
| Precision-Recall (Max Softmax) | ![PR Softmax](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__precision_recall_curve_max_softmax.png) |
| ROC Curve (Max Logit) | ![ROC Logit](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__roc_curve_max_logit.png) |
| ROC Curve (Max Softmax) | ![ROC Softmax](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__roc_curve_max_softmax.png) |
| Score Histogram (Max Logit) | ![Score Hist Logit](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000_score_hist_max_logit.png) |
| Score Histogram (Max Softmax) | ![Score Hist Softmax](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000_score_hist_max_softmax.png) |
| Score (Max Logit) | ![Score Max Logit](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__score_max_logit.png) |
| Score (Max Softmax) | ![Score Max Softmax](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__score_max_softmax.png) |
| Softmax Fake Data | ![Softmax Fake](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__softmax_Fake_data.png) |
| Softmax Real Data | ![Softmax Real](plots/es1e2/CNN_0.1/CNN_0.1_temp:1000__softmax_Real_data.png) |


</details>

</details>

<details>

<summary><strong>📊 CNN with esp=random(0.01-0.15) FGSM
training</strong></summary>

<br>

|  |  |
|---------|----------|
| Confusion Matrix | ![Confusion Matrix](plots/es1e2/CNN_None/CNN_None_temp:1000_confusion_matrix.png) |
| Input Fake Data | ![Input Fake](plots/es1e2/CNN_None/CNN_None_temp:1000__input_Fake_data.png) |
| Input Real Data | ![Input Real](plots/es1e2/CNN_None/CNN_None_temp:1000__input_Real_data.png) |
| Logit Fake Data | ![Logit Fake](plots/es1e2/CNN_None/CNN_None_temp:1000__logit_Fake_data.png) |
| Logit Real Data | ![Logit Real](plots/es1e2/CNN_None/CNN_None_temp:1000__logit_Real_data.png) |
| Precision-Recall (Max Logit) | ![PR Logit](plots/es1e2/CNN_None/CNN_None_temp:1000__precision_recall_curve_max_logit.png) |
| Precision-Recall (Max Softmax) | ![PR Softmax](plots/es1e2/CNN_None/CNN_None_temp:1000__precision_recall_curve_max_softmax.png) |
| ROC Curve (Max Logit) | ![ROC Logit](plots/es1e2/CNN_None/CNN_None_temp:1000__roc_curve_max_logit.png) |
| ROC Curve (Max Softmax) | ![ROC Softmax](plots/es1e2/CNN_None/CNN_None_temp:1000__roc_curve_max_softmax.png) |
| Score Histogram (Max Logit) | ![Score Hist Logit](plots/es1e2/CNN_None/CNN_None_temp:1000_score_hist_max_logit.png) |
| Score Histogram (Max Softmax) | ![Score Hist Softmax](plots/es1e2/CNN_None/CNN_None_temp:1000_score_hist_max_softmax.png) |
| Score (Max Logit) | ![Score Max Logit](plots/es1e2/CNN_None/CNN_None_temp:1000__score_max_logit.png) |
| Score (Max Softmax) | ![Score Max Softmax](plots/es1e2/CNN_None/CNN_None_temp:1000__score_max_softmax.png) |
| Softmax Fake Data | ![Softmax Fake](plots/es1e2/CNN_None/CNN_None_temp:1000__softmax_Fake_data.png) |
| Softmax Real Data | ![Softmax Real](plots/es1e2/CNN_None/CNN_None_temp:1000__softmax_Real_data.png) |

</details>

The histograms indicate that the baseline model struggles to clearly
differentiate between real and fake data. However, its performance
noticeably improves when FGSM is incorporated as a data augmentation
technique during training.

Using FGSM in this way improves OOD (Out-of-Distribution) detection. I
tested the model using small epsilon values, given the nature of
CIFAR-10.

Despite this improvement, OOD detection remains challenging. The
histograms reveal that the distributions of real and fake data still
partially overlap, meaning complete separation has not been achieved.

Comparing scoring functions, there is no clear or consistent advantage
in using either max_logit or max_softmax (with temperature fixed at
1000). In certain cases, one metric slightly outperforms the other, and
vice versa, as illustrated in the plots.

This model also appears less stable than the CNNplus model, both in
terms of OOD detection and during the training process, as reflected in
the results. Both models were trained with the Adam optimizer (learning
rate 0.0001) and a cosine annealing scheduler.

#### CNNplus

In terms of raw test set accuracy, the CNNplus model clearly outperforms
CNN, as reflected in the confusion matrices. It also demonstrates
superior performance on ROC and Precision-Recall curves compared to CNN.

The histograms show that the baseline model struggles to differentiate
between real and fake data. However, its performance improves when FGSM
is employed as a data augmentation technique during training.

Using FGSM in this way improves OOD (Out-of-Distribution) detection. I
tested the model using small epsilon values, given the nature of
CIFAR-10.

Overall, the max_softmax score (with temperature fixed at 1000)
generally yields better results than using raw logits directly.

There is no consistent advantage between max_logit and max_softmax as
scoring functions—each can outperform the other in specific cases, as
shown in the plots.

<details>

<summary><strong>📊 CNNplus with no FGSM training</strong></summary>

<br>

|  |  |
|---------|----------|
| Confusion Matrix | ![Confusion Matrix](plots/es1e2/CNNplus/CNNplus_temp:1000_confusion_matrix.png) |
| Input Fake Data | ![Input Fake](plots/es1e2/CNNplus/CNNplus_temp:1000__input_Fake_data.png) |
| Input Real Data | ![Input Real](plots/es1e2/CNNplus/CNNplus_temp:1000__input_Real_data.png) |
| Logit Fake Data | ![Logit Fake](plots/es1e2/CNNplus/CNNplus_temp:1000__logit_Fake_data.png) |
| Logit Real Data | ![Logit Real](plots/es1e2/CNNplus/CNNplus_temp:1000__logit_Real_data.png) |
| Precision-Recall (Max Logit) | ![PR Logit](plots/es1e2/CNNplus/CNNplus_temp:1000__precision_recall_curve_max_logit.png) |
| Precision-Recall (Max Softmax) | ![PR Softmax](plots/es1e2/CNNplus/CNNplus_temp:1000__precision_recall_curve_max_softmax.png) |
| ROC Curve (Max Logit) | ![ROC Logit](plots/es1e2/CNNplus/CNNplus_temp:1000__roc_curve_max_logit.png) |
| ROC Curve (Max Softmax) | ![ROC Softmax](plots/es1e2/CNNplus/CNNplus_temp:1000__roc_curve_max_softmax.png) |
| Score Histogram (Max Logit) | ![Score Hist Logit](plots/es1e2/CNNplus/CNNplus_temp:1000_score_hist_max_logit.png) |
| Score Histogram (Max Softmax) | ![Score Hist Softmax](plots/es1e2/CNNplus/CNNplus_temp:1000_score_hist_max_softmax.png) |
| Score (Max Logit) | ![Score Max Logit](plots/es1e2/CNNplus/CNNplus_temp:1000__score_max_logit.png) |
| Score (Max Softmax) | ![Score Max Softmax](plots/es1e2/CNNplus/CNNplus_temp:1000__score_max_softmax.png) |
| Softmax Fake Data | ![Softmax Fake](plots/es1e2/CNNplus/CNNplus_temp:1000__softmax_Fake_data.png) |
| Softmax Real Data | ![Softmax Real](plots/es1e2/CNNplus/CNNplus_temp:1000__softmax_Real_data.png) |
</details>

<details>

<summary><strong>📊 CNNplus with eps=0.05 FGSM
training</strong></summary>

<br>

|  |  |
|---------|----------|
| Confusion Matrix | ![Confusion Matrix](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000_confusion_matrix.png) |
| Input Fake Data | ![Input Fake](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__input_Fake_data.png) |
| Input Real Data | ![Input Real](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__input_Real_data.png) |
| Logit Fake Data | ![Logit Fake](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__logit_Fake_data.png) |
| Logit Real Data | ![Logit Real](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__logit_Real_data.png) |
| Precision-Recall (Max Logit) | ![PR Logit](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__precision_recall_curve_max_logit.png) |
| Precision-Recall (Max Softmax) | ![PR Softmax](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__precision_recall_curve_max_softmax.png) |
| ROC Curve (Max Logit) | ![ROC Logit](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__roc_curve_max_logit.png) |
| ROC Curve (Max Softmax) | ![ROC Softmax](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__roc_curve_max_softmax.png) |
| Score Histogram (Max Logit) | ![Score Hist Logit](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000_score_hist_max_logit.png) |
| Score Histogram (Max Softmax) | ![Score Hist Softmax](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000_score_hist_max_softmax.png) |
| Score (Max Logit) | ![Score Max Logit](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__score_max_logit.png) |
| Score (Max Softmax) | ![Score Max Softmax](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__score_max_softmax.png) |
| Softmax Fake Data | ![Softmax Fake](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__softmax_Fake_data.png) |
| Softmax Real Data | ![Softmax Real](plots/es1e2/CNNplus_0.05/CNNplus_0.05_temp:1000__softmax_Real_data.png) |

</details>

</details>

<details>

<summary><strong>📊 CNNplus with eps=0.1 FGSM
training</strong></summary>

<br>

|  |  |
|---------|----------|
| Confusion Matrix | ![Confusion Matrix](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000_confusion_matrix.png) |
| Input Fake Data | ![Input Fake](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__input_Fake_data.png) |
| Input Real Data | ![Input Real](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__input_Real_data.png) |
| Logit Fake Data | ![Logit Fake](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__logit_Fake_data.png) |
| Logit Real Data | ![Logit Real](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__logit_Real_data.png) |
| Precision-Recall (Max Logit) | ![PR Logit](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__precision_recall_curve_max_logit.png) |
| Precision-Recall (Max Softmax) | ![PR Softmax](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__precision_recall_curve_max_softmax.png) |
| ROC Curve (Max Logit) | ![ROC Logit](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__roc_curve_max_logit.png) |
| ROC Curve (Max Softmax) | ![ROC Softmax](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__roc_curve_max_softmax.png) |
| Score Histogram (Max Logit) | ![Score Hist Logit](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000_score_hist_max_logit.png) |
| Score Histogram (Max Softmax) | ![Score Hist Softmax](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000_score_hist_max_softmax.png) |
| Score (Max Logit) | ![Score Max Logit](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__score_max_logit.png) |
| Score (Max Softmax) | ![Score Max Softmax](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__score_max_softmax.png) |
| Softmax Fake Data | ![Softmax Fake](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__softmax_Fake_data.png) |
| Softmax Real Data | ![Softmax Real](plots/es1e2/CNNplus_0.1/CNNplus_0.1_temp:1000__softmax_Real_data.png) |


</details>

</details>

<details>

<summary><strong>📊 CNNplus with esp=random(0.01-0.15) FGSM
training</strong></summary>

<br>

|  |  |
|---------|----------|
| Confusion Matrix | ![Confusion Matrix](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000_confusion_matrix.png) |
| Input Fake Data | ![Input Fake](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__input_Fake_data.png) |
| Input Real Data | ![Input Real](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__input_Real_data.png) |
| Logit Fake Data | ![Logit Fake](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__logit_Fake_data.png) |
| Logit Real Data | ![Logit Real](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__logit_Real_data.png) |
| Precision-Recall (Max Logit) | ![PR Logit](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__precision_recall_curve_max_logit.png) |
| Precision-Recall (Max Softmax) | ![PR Softmax](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__precision_recall_curve_max_softmax.png) |
| ROC Curve (Max Logit) | ![ROC Logit](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__roc_curve_max_logit.png) |
| ROC Curve (Max Softmax) | ![ROC Softmax](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__roc_curve_max_softmax.png) |
| Score Histogram (Max Logit) | ![Score Hist Logit](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000_score_hist_max_logit.png) |
| Score Histogram (Max Softmax) | ![Score Hist Softmax](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000_score_hist_max_softmax.png) |
| Score (Max Logit) | ![Score Max Logit](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__score_max_logit.png) |
| Score (Max Softmax) | ![Score Max Softmax](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__score_max_softmax.png) |
| Softmax Fake Data | ![Softmax Fake](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__softmax_Fake_data.png) |
| Softmax Real Data | ![Softmax Real](plots/es1e2/CNNplus_None/CNNplus_None_temp:1000__softmax_Real_data.png) |

</details>

#### Autoencoder

In general, the AutoEncoder model is more robust and better suited for
anomaly detection, as shown by the plots, especially the scores in the
histogram, which clearly highlight this.

Training the model with FGSM as a data augmentation technique sometimes
leads to slightly better performance, but the improvement is marginal in
the context of this experiment.

In all cases, the network appears to detect the difference between real
and fake data much more effectively than CNN-based models.

<details>

<summary><strong>📊 Autoencoder with no FGSM training</strong></summary>

<br>

|  |  |
|---------|----------|
| Precision-Recall Curve | ![Precision-Recall](plots/es1e2/Autoencoder/Autoencoder_temp:1000__precision_recall_curve_reconstruction_error.png) |
| ROC Curve | ![ROC](plots/es1e2/Autoencoder/Autoencoder_temp:1000__roc_curve_reconstruction_error.png) |
| Score Histogram | ![Score Hist](plots/es1e2/Autoencoder/Autoencoder_temp:1000_score_hist_reconstruction_error.png) |
| Score Reconstruction Error | ![Score RE](plots/es1e2/Autoencoder/Autoencoder_temp:1000__score_reconstruction_error.png) |

</details>

<details>

<summary><strong>📊 Autoencoder with eps=0.05 FGSM
training</strong></summary>

<br>

|  |  |
|---------|----------|
| Precision-Recall Curve | ![Precision-Recall](plots/es1e2/Autoencoder_0.05/Autoencoder_0.05_temp:1000__precision_recall_curve_reconstruction_error.png) |
| ROC Curve | ![ROC](plots/es1e2/Autoencoder_0.05/Autoencoder_0.05_temp:1000__roc_curve_reconstruction_error.png) |
| Score Histogram | ![Score Hist](plots/es1e2/Autoencoder_0.05/Autoencoder_0.05_temp:1000_score_hist_reconstruction_error.png) |
| Score Reconstruction Error | ![Score RE](plots/es1e2/Autoencoder_0.05/Autoencoder_0.05_temp:1000__score_reconstruction_error.png) |


</details>

</details>

<details>

<summary><strong>📊 Autoencoder with eps=0.1 FGSM
training</strong></summary>

<br>

|  |  |
|---------|----------|
| Precision-Recall Curve | ![Precision-Recall](plots/es1e2/Autoencoder_0.1/Autoencoder_0.1_temp:1000__precision_recall_curve_reconstruction_error.png) |
| ROC Curve | ![ROC](plots/es1e2/Autoencoder_0.1/Autoencoder_0.1_temp:1000__roc_curve_reconstruction_error.png) |
| Score Histogram | ![Score Hist](plots/es1e2/Autoencoder_0.1/Autoencoder_0.1_temp:1000_score_hist_reconstruction_error.png) |
| Score Reconstruction Error | ![Score RE](plots/es1e2/Autoencoder_0.1/Autoencoder_0.1_temp:1000__score_reconstruction_error.png) |


</details>

</details>

<details>

<summary><strong>📊 Autoencoder with esp=random(0.01-0.15) FGSM
training</strong></summary>

<br>

|  |  |
|---------|----------|
| Precision-Recall Curve | ![Precision-Recall](plots/es1e2/Autoencoder_None/Autoencoder_None_temp:1000__precision_recall_curve_reconstruction_error.png) |
| ROC Curve | ![ROC](plots/es1e2/Autoencoder_None/Autoencoder_None_temp:1000__roc_curve_reconstruction_error.png) |
| Score Histogram | ![Score Hist](plots/es1e2/Autoencoder_None/Autoencoder_None_temp:1000_score_hist_reconstruction_error.png) |
| Score Reconstruction Error | ![Score RE](plots/es1e2/Autoencoder_None/Autoencoder_None_temp:1000__score_reconstruction_error.png) |

</details>

### Results Exercise 2

These exercises implement the FGSM Attack based on [PyTorch's FGSM
tutorial](https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html).

For exercise 2.2, the implementation follows the training approach
described in ["Training Augmentation with Adversarial Examples for
Robust Speech Recognition"](https://arxiv.org/abs/1806.02782), testing
with these epsilon values: [0.0, 0.05, 0.075, 0.1, 0.125, 0.15].

#### CNN

When trained normally (without adversarial augmentation), the model's
accuracy drops immediately to 0% under an FGSM attack at any tested
epsilon value. However, when trained with FGSM as a data augmentation
technique, the model becomes significantly more robust to such attacks.

The best performance is observed when a random epsilon between 0.01 and
0.15 is used for FGSM during training.

|Epsilon     |Loss     |Examples    |
|-----|-----|-----|
| No FGSM   | ![FGSM](plots/es1e2/CNN/FGSM_eps_CNN.png) | ![FGSM Example](plots/es1e2/CNN/FGSM_EXAMPLE_IMG_CNN.png) |
| FGSM eps=0.05| ![FGSM](plots/es1e2/CNN_0.05/FGSM_eps_CNN.png) | ![FGSM Example](plots/es1e2/CNN_0.05/FGSM_EXAMPLE_IMG_CNN.png) |
| FGSM eps=0.1| ![FGSM](plots/es1e2/CNN_0.1/FGSM_eps_CNN.png?v2) | ![FGSM Example](plots/es1e2/CNN_0.1/FGSM_EXAMPLE_IMG_CNN.png?v2) |
| FGSM eps=random| ![FGSM](plots/es1e2/CNN_None/FGSM_eps_CNN.png) | ![FGSM Example](plots/es1e2/CNN_None/FGSM_EXAMPLE_IMG_CNN.png) |

#### CNNplus

When trained normally (without adversarial augmentation), the model's
accuracy drops immediately to 0% under an FGSM attack at any tested
epsilon value. However, when trained with FGSM as a data augmentation
technique, the model becomes significantly more robust to such attacks.

The best performance is observed when a random epsilon between 0.01 and
0.15 is used for FGSM during training. in this case, the slightly lower
accuracy at small epsilon values may be due to the fact that the epsilon
value lies near the edge of the uniform distribution `[0.01, 0.15]` used
during training. As a result, the model may have seen fewer examples
with that perturbation, reducing its accuracy in that region.

|Epsilon     |Loss     |Examples    |
|-----|-----|-----|
| No FGSM   | ![FGSM](plots/es1e2/CNNplus/FGSM_eps_CNN.png) | ![FGSM Example](plots/es1e2/CNNplus/FGSM_EXAMPLE_IMG_CNN.png) |
| FGSM eps=0.05| ![FGSM](plots/es1e2/CNNplus_0.05/FGSM_eps_CNN.png) | ![FGSM Example](plots/es1e2/CNNplus_0.05/FGSM_EXAMPLE_IMG_CNN.png) |
| FGSM eps=0.1| ![FGSM](plots/es1e2/CNNplus_0.1/FGSM_eps_CNN.png) | ![FGSM Example](plots/es1e2/CNNplus_0.1/FGSM_EXAMPLE_IMG_CNN.png) |
| FGSM eps=random| ![FGSM](plots/es1e2/CNNplus_None/FGSM_eps_CNN.png) | ![FGSM Example](plots/es1e2/CNNplus_None/FGSM_EXAMPLE_IMG_CNN.png) |

#### Autoencoder

In terms of reconstruction loss (measured by MSE Loss), the autoencoder
model performs better when FGSM is used as a data augmentation technique
during training, with a random epsilon sampled between 0.01 and 0.15.

In general the Autoencoder models seems more robust than the CNN and
CNNplus model at this type of attack.

|Epsilon     |Loss     |Examples    |
|-----|-----|-----|
| No FGSM   | ![FGSM eps](plots/es1e2/Autoencoder/FGSM_eps_Autoencoder.png) | ![FGSM Example](plots/es1e2/Autoencoder/FGSM_EXAMPLE_IMG_Autoencoder.png) |
| FGSM eps=0.05| ![FGSM eps](plots/es1e2/Autoencoder_0.05/FGSM_eps_Autoencoder.png) | ![FGSM Example](plots/es1e2/Autoencoder_0.05/FGSM_EXAMPLE_IMG_Autoencoder.png) |
| FGSM eps=0.1| ![FGSM eps](plots/es1e2/Autoencoder_0.1/FGSM_eps_Autoencoder.png?v2) | ![FGSM Example](plots/es1e2/Autoencoder_0.1/FGSM_EXAMPLE_IMG_Autoencoder.png) |
| FGSM eps=random| ![FGSM eps](plots/es1e2/Autoencoder_None/FGSM_eps_Autoencoder.png?v2) | ![FGSM Example](plots/es1e2/Autoencoder_None/FGSM_EXAMPLE_IMG_Autoencoder.png) |

# Experiment 3

### Parameters

To run this experiment use:

```         
python main.py --experiment 3
```

For these experiment I use this configuration, that can be found in
`/configs/config_3.yaml`:

```         
seed: 99
device: auto

data:
  batch_size: 256
  validation_split: 10
  num_workers: 2
  mean: [0.4914, 0.4822, 0.4465]
  std: [0.2023, 0.1994, 0.2010]

#Configurazione per l'esperimento 3
models:
  #Lista dei modelli da testare
  cnn_models:
  
    - name: "CNN"
      path: "models/CNN.pth"
      
    - name: "CNNplus"
      path: "models/CNNplus.pth"
      
    - name: "CNN_0.05"
      path: "models/CNN_0.05.pth"
      
    - name: "CNNplus_0.05"
      path: "models/CNNplus_0.05.pth"
      
    - name: "CNN_0.1"
      path: "models/CNN_0.1.pth"
      
    - name: "CNNplus_0.1"
      path: "models/CNNplus_0.1.pth"
      
    - name: "CNN_None"
      path: "models/CNN_None.pth"
      
    - name: "CNNplus_None"
      path: "models/CNNplus_None.pth"
      
    
fgsm:
  epsilons_cnn: [0.0, 0.05, 0.075, 0.1, 0.125, 0.15]
  epsilons_ae: [0.0, 0.05, 0.075, 0.1, 0.125, 0.15]
  target_class: 0

logging:
  project_name: "Lab4-OOD_Detection"
```

### What and How

In the third exercise, the objective was to implement a targeted FGSM
attack and analyze the results both quantitatively and qualitatively.

For the implementation, I used the code written for the non-targeted
FGSM attack and slightly modified it so that, in examples where the
model did not already predict the target class, it would instead be
pushed to predict it. The key was to apply two small adjustments: the
first was to replace the original label with the target class, so that
the loss computed during the forward pass measures how far the model is
from predicting the target class; the second was to pass `-epsilon` to
the FGSM attack method. With these adjustments, the attack no longer
maximizes the distance from the original prediction but instead
minimizes the distance to the target.

To quantitatively analyze the performance, I used the following metrics.

```         
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
```
A summary of the performance can be seen in here ![output.txt](https://github.com/coseemo/DLA_LABS/blob/main/lab4/output.txt)

For the qualitative analysis, I plotted figures showing the original
image, the applied perturbation, and the adversarial image, highlighting
the cases in which the attack was successful and those in which it was
not.

### Results

In general, we observe that attacks become more effective as epsilon increases for models that are not trained with adversarial training. Conversely, for models trained with adversarial training, the attacks tend to be more effective at lower epsilon values, a behavior that is consistent across both the CNN and the CNNplus models.
It is also interesting to note how the nature of the perturbations changes depending on both the epsilon used during adversarial training and the epsilon used during testing. In particular, when a higher epsilon is employed during adversarial training, the resulting perturbations appear to lose their structured form, whereas lower training epsilon values lead to perturbations that more closely resemble the shape of an image belonging to the target class. On the other hand, the epsilon used during testing seems to push the perturbation toward regions of the image characterized by high contrast (such as edges?).
Furthermore, when high epsilon values are used both during training and testing, the attack consistently fails: in these cases, the perturbed image appears visually very similar to the original one, differing only in some color variations. In contrast, successful attacks tend to occur when the perturbation is not “constrained” by the model to lose the original target-class structure, to the extent that the adversarial image appears as a superposition of the original image and an image belonging to the target class.

#### CNN

|     |     |
|-----|-----|
| ![FGSM eps](plots/es3/CNN/FGSM_SUCCESS_RATE_TARGET_0_.png) |  ![FGSM eps](plots/es3/CNN_0.05/FGSM_SUCCESS_RATE_TARGET_0_.png)|
|  ![FGSM eps](plots/es3/CNN_0.1/FGSM_SUCCESS_RATE_TARGET_0_.png) |  ![FGSM eps](plots/es3/CNN_None/FGSM_SUCCESS_RATE_TARGET_0_.png) |

<details>

<summary><strong>📊 Example for CNN with no FGSM training</strong></summary>

<br>

| Epsilon | Examples |
|--------|----------|
| FGSM eps = 0.0 | ![eps 0.0](plots/es3/CNN/targeted_examples_class0_CNN_eps0.0.png) |
| FGSM eps = 0.05 | ![eps 0.05](plots/es3/CNN/targeted_examples_class0_CNN_eps0.05.png) |
| FGSM eps = 0.075 | ![eps 0.075](plots/es3/CNN/targeted_examples_class0_CNN_eps0.075.png) |
| FGSM eps = 0.1 | ![eps 0.1](plots/es3/CNN/targeted_examples_class0_CNN_eps0.1.png) |
| FGSM eps = 0.125 | ![eps 0.125](plots/es3/CNN/targeted_examples_class0_CNN_eps0.125.png) |
| FGSM eps = 0.15 | ![eps 0.15](plots/es3/CNN/targeted_examples_class0_CNN_eps0.15.png) |

</details>

<details>

<summary><strong>📊 Example for CNN with FGSM training eps=0.05</strong></summary>

<br>

| Epsilon | Examples |
|--------|----------|
| FGSM eps = 0.0 | ![eps 0.0](plots/es3/CNN_0.05/targeted_examples_class0_CNN_0.05_eps0.0.png) |
| FGSM eps = 0.05 | ![eps 0.05](plots/es3/CNN_0.05/targeted_examples_class0_CNN_0.05_eps0.05.png) |
| FGSM eps = 0.075 | ![eps 0.075](plots/es3/CNN_0.05/targeted_examples_class0_CNN_0.05_eps0.075.png) |
| FGSM eps = 0.1 | ![eps 0.1](plots/es3/CNN_0.05/targeted_examples_class0_CNN_0.05_eps0.1.png) |
| FGSM eps = 0.125 | ![eps 0.125](plots/es3/CNN_0.05/targeted_examples_class0_CNN_0.05_eps0.125.png) |
| FGSM eps = 0.15 | ![eps 0.15](plots/es3/CNN_0.05/targeted_examples_class0_CNN_0.05_eps0.15.png) |


</details>

<details>

<summary><strong>📊 Example for CNN with FGSM training eps=0.1</strong></summary>

<br>

| Epsilon | Examples |
|--------|----------|
| FGSM eps = 0.0 | ![eps 0.0](plots/es3/CNN_0.1/targeted_examples_class0_CNN_0.1_eps0.0.png) |
| FGSM eps = 0.05 | ![eps 0.1](plots/es3/CNN_0.1/targeted_examples_class0_CNN_0.1_eps0.05.png) |
| FGSM eps = 0.075 | ![eps 0.075](plots/es3/CNN_0.1/targeted_examples_class0_CNN_0.1_eps0.075.png) |
| FGSM eps = 0.1 | ![eps 0.1](plots/es3/CNN_0.1/targeted_examples_class0_CNN_0.1_eps0.1.png) |
| FGSM eps = 0.125 | ![eps 0.125](plots/es3/CNN_0.1/targeted_examples_class0_CNN_0.1_eps0.125.png) |
| FGSM eps = 0.15 | ![eps 0.15](plots/es3/CNN_0.1/targeted_examples_class0_CNN_0.1_eps0.15.png) |

</details>

<details>

<summary><strong>📊 Example for CNN with FGSM training eps=Random</strong></summary>

<br>

| Epsilon | Examples |
|--------|----------|
| FGSM eps = 0.0 | ![eps 0.0](plots/es3/CNN_None/targeted_examples_class0_CNN_None_eps0.0.png) |
| FGSM eps = 0.05 | ![eps 0.1](plots/es3/CNN_None/targeted_examples_class0_CNN_None_eps0.05.png) |
| FGSM eps = 0.075 | ![eps 0.075](plots/es3/CNN_None/targeted_examples_class0_CNN_None_eps0.075.png) |
| FGSM eps = 0.1 | ![eps 0.1](plots/es3/CNN_None/targeted_examples_class0_CNN_None_eps0.1.png) |
| FGSM eps = 0.125 | ![eps 0.125](plots/es3/CNN_None/targeted_examples_class0_CNN_None_eps0.125.png) |
| FGSM eps = 0.15 | ![eps 0.15](plots/es3/CNN_None/targeted_examples_class0_CNN_None_eps0.15.png) |


</details>



#### CNNplus

|     |     |
|-----|-----|
| ![FGSM eps](plots/es3/CNNplus/FGSM_SUCCESS_RATE_TARGET_0_.png) |  ![FGSM eps](plots/es3/CNNplus_0.05/FGSM_SUCCESS_RATE_TARGET_0_.png)|
|  ![FGSM eps](plots/es3/CNNplus_0.1/FGSM_SUCCESS_RATE_TARGET_0_.png) |  ![FGSM eps](plots/es3/CNNplus_None/FGSM_SUCCESS_RATE_TARGET_0_.png) |

|     |     |
|-----|-----|
| ![FGSM eps](plots/es3/CNN/FGSM_SUCCESS_RATE_TARGET_0_.png) |  ![FGSM eps](plots/es3/CNN_0.05/FGSM_SUCCESS_RATE_TARGET_0_.png)|
|  ![FGSM eps](plots/es3/CNN_0.1/FGSM_SUCCESS_RATE_TARGET_0_.png) |  ![FGSM eps](plots/es3/CNN_None/FGSM_SUCCESS_RATE_TARGET_0_.png) |

<details>

<summary><strong>📊 Example for CNNplus with no FGSM training</strong></summary>

<br>

| Epsilon | Examples |
|--------|----------|
| FGSM eps = 0.0 | ![eps 0.0](plots/es3/CNNplus/targeted_examples_class0_CNNplus_eps0.0.png) |
| FGSM eps = 0.05 | ![eps 0.05](plots/es3/CNNplus/targeted_examples_class0_CNNplus_eps0.05.png) |
| FGSM eps = 0.075 | ![eps 0.075](plots/es3/CNNplus/targeted_examples_class0_CNNplus_eps0.075.png) |
| FGSM eps = 0.1 | ![eps 0.1](plots/es3/CNNplus/targeted_examples_class0_CNNplus_eps0.1.png) |
| FGSM eps = 0.125 | ![eps 0.125](plots/es3/CNNplus/targeted_examples_class0_CNNplus_eps0.125.png) |
| FGSM eps = 0.15 | ![eps 0.15](plots/es3/CNNplus/targeted_examples_class0_CNNplus_eps0.15.png) |

</details>

<details>

<summary><strong>📊 Example for CNNplus with FGSM training eps=0.05</strong></summary>

<br>

| Epsilon | Examples |
|--------|----------|
| FGSM eps = 0.0 | ![eps 0.0](plots/es3/CNNplus_0.05/targeted_examples_class0_CNNplus_0.05_eps0.0.png) |
| FGSM eps = 0.05 | ![eps 0.05](plots/es3/CNNplus_0.05/targeted_examples_class0_CNNplus_0.05_eps0.05.png) |
| FGSM eps = 0.075 | ![eps 0.075](plots/es3/CNNplus_0.05/targeted_examples_class0_CNNplus_0.05_eps0.075.png) |
| FGSM eps = 0.1 | ![eps 0.1](plots/es3/CNNplus_0.05/targeted_examples_class0_CNNplus_0.05_eps0.1.png) |
| FGSM eps = 0.125 | ![eps 0.125](plots/es3/CNNplus_0.05/targeted_examples_class0_CNNplus_0.05_eps0.125.png) |
| FGSM eps = 0.15 | ![eps 0.15](plots/es3/CNNplus_0.05/targeted_examples_class0_CNNplus_0.05_eps0.15.png) |


</details>

<details>

<summary><strong>📊 Example for CNNplus with FGSM training eps=0.1</strong></summary>

<br>

| Epsilon | Examples |
|--------|----------|
| FGSM eps = 0.0 | ![eps 0.0](plots/es3/CNNplus_0.1/targeted_examples_class0_CNNplus_0.1_eps0.0.png) |
| FGSM eps = 0.05 | ![eps 0.1](plots/es3/CNNplus_0.1/targeted_examples_class0_CNNplus_0.1_eps0.05.png) |
| FGSM eps = 0.075 | ![eps 0.075](plots/es3/CNNplus_0.1/targeted_examples_class0_CNNplus_0.1_eps0.075.png) |
| FGSM eps = 0.1 | ![eps 0.1](plots/es3/CNNplus_0.1/targeted_examples_class0_CNNplus_0.1_eps0.1.png) |
| FGSM eps = 0.125 | ![eps 0.125](plots/es3/CNNplus_0.1/targeted_examples_class0_CNNplus_0.1_eps0.125.png) |
| FGSM eps = 0.15 | ![eps 0.15](plots/es3/CNNplus_0.1/targeted_examples_class0_CNNplus_0.1_eps0.15.png) |

</details>

<details>

<summary><strong>📊 Example for CNNplus with FGSM training eps=Random</strong></summary>

<br>

| Epsilon | Examples |
|--------|----------|
| FGSM eps = 0.0 | ![eps 0.0](plots/es3/CNNplus_None/targeted_examples_class0_CNNplus_None_eps0.0.png) |
| FGSM eps = 0.05 | ![eps 0.1](plots/es3/CNNplus_None/targeted_examples_class0_CNNplus_None_eps0.05.png) |
| FGSM eps = 0.075 | ![eps 0.075](plots/es3/CNNplus_None/targeted_examples_class0_CNNplus_None_eps0.075.png) |
| FGSM eps = 0.1 | ![eps 0.1](plots/es3/CNNplus_None/targeted_examples_class0_CNNplus_None_eps0.1.png) |
| FGSM eps = 0.125 | ![eps 0.125](plots/es3/CNNplus_None/targeted_examples_class0_CNNplus_None_eps0.125.png) |
| FGSM eps = 0.15 | ![eps 0.15](plots/es3/CNNplus_None/targeted_examples_class0_CNNplus_None_eps0.15.png) |


</details>
