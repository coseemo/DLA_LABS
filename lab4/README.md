
# Laboratory 4: OOD, FGSM, JARN


## Plots
All the plots can be found here:
- `plots`: []
- **lab4:** [https://wandb.ai/cosimo-borghini1-universit-di-firenze/LAB1-CNN?nw=nwusercosimoborghini1]


[If you expand the runs, you can see which parameters i used for each run]

## Pretraining
### Parameters
To run this experiment use:

    python main.py --experiment pretrain

For these experiment I use this configuration, that can be found in `/configs/config_pretrain.yaml`:
    
    seed: 99
    device: cuda
    
    data:
      batch_size: 128
      validation_split: 0.2
      num_workers: 4
      mean: [0.4914, 0.4822, 0.4465]
      std: [0.2023, 0.1994, 0.2010]
    
    model:
      path: "models/cifar10_CNN.pth"  #"models/cifar10_CNN.pth", "models/cifar10_CNNplus.pth", "models/cifar10_Autoencoder.pth"
    
    train:
      criterion: "CrossEntropyLoss"                #per le cnn "CrossEntropyLoss" per gli autoencoder "MSELoss"
      optimizer: "Adam"
      criterion_params: null                       #null per le cnn, {"reduction":"none"} per gli autoencoder
      lr: 0.001
          epochs: 200                      #cnn:200 cnnplus:50 autoencoder:200
          scheduler: "CosineAnnealingLR"
    
    logging:
      project_name: "Lab4-OOD_Detection"
### Results
-   **CNN**
    
    -   Simpler architecture
        
    -   Converges in ~200 epochs
        
    -   Uses Adam optimizer (lr=0.0001) with cosine annealing scheduler
        
    -   Achieves reasonable accuracy, but lower than CNNplus
    
    
|  | accuracy | loss
|--|--|--
| train | ![trainacc]() |![trainloss]()
| val |![valacc]() |![vallos]()
|test| ![testacc]()|![testloss]()
        
-   **CNNplus**
    
    -   More expressive and stable architecture
        
    -   Converges faster (~50 epochs)
        
    -   Uses Adam optimizer (lr=0.0001) with cosine annealing scheduler
        
    -   Achieves higher accuracy than CNN

|  | accuracy | loss
|--|--|--
| train | ![trainacc]() |![trainloss]()
| val |![valacc]() |![vallos]()
|test| ![testacc]()|![testloss]()
        
-   **AutoEncoder**
    
    -   Trained for 200 epochs
        
    -   Uses Adam optimizer (lr=0.0001) with cosine annealing scheduler
        
    -   Loss function: Mean Squared Error (MSELoss) on reconstruction output

|  | accuracy | loss
|--|--|--
| train | ![trainacc]() |![trainloss]()
| val |![valacc]() |![vallos]()
|test| ![testacc]()|![testloss]()

## Experiment 1
### Parameters
To run this experiment use:

    python main.py --experiment 1

For these experiment I use this configuration, that can be found in `/configs/config_1.yaml`:

 

    seed: 6
    
    device: auto
    
    data:
      batch_size: 128
      validation_split: 0.2
      num_workers: 4
      mean: [0.4914, 0.4822, 0.4465]
      std: [0.2023, 0.1994, 0.2010]
    
    fake_data:
      batch_size: 128
      num_workers: 4
    
    model:
      path: "models/Autoencoder.pth"           #"models/CNN.pth", "models/CNNplus.pth", "models/Autoencoder.pth"
      temps: [1000.0]                   #test con più temperature
    
    train:
      criterion: "MSELoss"            #per le cnn "CrossEntropyLoss" per gli autoencoder "MSELoss"
      criterion_params: {"reduction":"none"}  #per le cnn null mentre per gli autoencoder {"reduction":"none"}
    
    logging:
      project_name: "Lab4-OOD_Detection"


The commented parameters are the ones used for the various runs.

## Experiment 2
### Parameters
To run this experiment use:

    python main.py --experiment 2

For these experiment I use this configuration, that can be found in `/configs/config_2.yaml`:

    seed: 99
    
    device: auto
    
    data:
      batch_size: 128
      validation_split: 20
      num_workers: 2
      mean: [0.4914, 0.4822, 0.4465]
      std: [0.2023, 0.1994, 0.2010]
    
    model:
      path: "models/Autoencoder.pth"          #"models/cifar10_CNN.pth", "models/cifar10_CNNplus.pth", "models/cifar10_Autoencoder.pth"
      
    setup:
      criterion: "MSELoss"           #per le cnn "CrossEntropyLoss" per gli autoencoder "MSELoss"
      optimizer: "Adam"
      criterion_params: {"reduction":"none"}                       #null per le cnn, {"reduction":"none"} per gli autoencoder
      lr: 0.0001
      epochs: 200                          #cnnplus:50, cnn:200, autoencoder:200
      scheduler: "CosineAnnealingLR"
    
    fgsm:
      epsilons_cnn: [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
      epsilons_ae: [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
      train_epsilon: null                  #si può provare con epsilon fissato ed epsilon casuale                
    
    logging:
      project_name: "Lab4-OOD_Detection"

### Results 1&2

#### CNN
The histograms indicate that the baseline model struggles to clearly differentiate between real and fake data. However, its performance noticeably improves when FGSM is incorporated as a data augmentation technique during training.

Using FGSM in this way improves OOD (Out-of-Distribution) detection. I tested the model using small epsilon values, given the nature of CIFAR-10, and to facilitate comparison with the experiments using JARN

Despite this improvement, OOD detection remains challenging. The histograms reveal that the distributions of real and fake data still partially overlap, meaning complete separation has not been achieved.

Comparing scoring functions, there is no clear or consistent advantage in using either max_logit or max_softmax (with temperature fixed at 1000). In certain cases, one metric slightly outperforms the other, and vice versa, as illustrated in the plots.

This model also appears less stable than the CNNplus model, both in terms of OOD detection and during the training process, as reflected in the results. Both models were trained with the Adam optimizer (learning rate 0.0001) and a cosine annealing scheduler.

|  | baseline | trained|
|--|--|--|
| roc | ![rcb]() |![rct]() |
|precision-recall|![prb]() |![prt]() |
|histogram|![hb]() |![ht]() |
|score|![sb]() |![st]() |
| conf | ![cb]() |![ct]() |
|mse|![mb]() |![mt]() |



#### CNNplus

In terms of raw test set accuracy, the CNNplus model clearly outperforms CNN, as reflected in the confusion matrices. It also demonstrates superior performance on ROC and Precision-Recall curves compared to CNN.

The histograms show that the baseline model struggles to differentiate between real and fake data. However, its performance improves when FGSM is employed as a data augmentation technique during training.

Using FGSM in this way improves OOD (Out-of-Distribution) detection. I tested the model using small epsilon values, given the nature of CIFAR-10, and to facilitate comparison with the experiments using JARN

Overall, the max_softmax score (with temperature fixed at 1000) generally yields better results than using raw logits directly.

That said, there is no consistent advantage between max_logit and max_softmax as scoring functions—each can outperform the other in specific cases, as shown in the plots.

|  | baseline | trained|
|--|--|--|
| roc | ![rcb]() |![rct]() |
|precision-recall|![prb]() |![prt]() |
|histogram|![hb]() |![ht]() |
|score|![sb]() |![st]() |
| conf | ![cb]() |![ct]() |
|mse|![mb]() |![mt]() |


#### Autoencoder

In general, the AutoEncoder model is more robust and better suited for anomaly detection, as shown by the plots, especially the scores in the histogram, which clearly highlight this.

Training the model with FGSM as a data augmentation technique sometimes leads to slightly better performance, but the improvement is marginal in the context of this experiment.

In all cases, the network appears to detect the difference between real and fake data much more effectively than CNN-based models.
|  | baseline | trained|
|--|--|--|
| roc | ![rcb]() |![rct]() |
|precision-recall|![prb]() |![prt]() |
|histogram|![hb]() |![ht]() |
|score|![sb]() |![st]() |

## Experiment 3
### Parameters
To run this experiment use:

    python main.py --experiment 3

For these experiment I use this configuration, that can be found in `/configs/config_3.yaml`:

    seed: 99
    
    device: auto
    
    data:
      mean: [0.4914, 0.4822, 0.4465]
      std: [0.2023, 0.1994, 0.2010]
    
    training:
      lr: 0.001
      epochs: 50
    
    fgsm:
      epsilons: [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    
    jarn:
      adv_weight: 1                   
      epsilon: 0.031                     
      disc_opt_lr: 0.0072
      start_ratio: 0.75                  #a che percentuale si inizia a utilizzare jarn
    
    logging:
      project_name: "Lab4-OOD_Detection"

In the third exercise, the goal was to implement the JARN method from the paper [https://arxiv.org/abs/1912.10185](https://arxiv.org/abs/1912.10185?utm_source=chatgpt.com)

JARN is designed to improve neural network robustness against adversarial attacks by incorporating **Jacobian-based regularization**. The method combines:

-  **Standard training** – optimizing the network on the true labels.
    
-  **Adversarial training** – including perturbed examples (such as FGSM) to train the model to resist small input perturbations 
-   **Combined loss**  - we use a combination of the standard and adversarial loss for improve our model.

I used the hyperparameters of the paper and what is observed is a slight increase in the model's resilience: indeed, while the accuracy of the baseline model and the model trained with FGSM drops quickly, with JARN this behavior is mitigated, leading to a slower decrease, as can be seen in the graphs.
|  | baseline | fgsm| jarn|
|--|--|--| --|
|mse|![mb]() |![mf]() |![mj]()



