
# Laboratory 1: MLP, CNN, Residual Conections and Distillation


## Plots
All the plots can be found here:
- **lab1:** [https://wandb.ai/cosimo-borghini1-universit-di-firenze/LAB1-CNN?nw=nwusercosimoborghini1]

[If you expand the runs, you can see which parameters i used for each run]

## Experiment 1.1
### Parameters
To run this experiment use:

    python main.py --experiment 1_1

For these experiment I use this configuration, that can be found in `/configs/config_exp1_1.yaml`:

 

        #Esperimento 1.1: MLP con MNIST
    seed: 42
    
    data:
      batch_size: 128
      validation_split: 20  #Percentuale che definisce la grandezza del validation set
      num_workers: 4
    
    model:
      type: "MLP"
      params:
        input_size: 784 #28*28 per MNIST
        layers_dim: 128  #32 #64 #128
        class_num: 10
        hidden_layers_num: 40 #10 #20 #40
        residual: false
        activation: "ReLU"
        dropout: 0.0
        batch_norm: true
    
    training:
      epochs: 50
      criterion: "CrossEntropyLoss"
      optimizer: "Adam"
      lr: 0.001
      scheduler: "CosineAnnealingLR"
    
    logging:
      project_name: "LAB1-CNN"
      log_gradients: false

The commented parameters are the ones used for the various runs.
### Results
The first exercise focused on testing the MLP implemented on the MNIST dataset, with particular attention to the variation in performance as the model’s depth and width increased. As we can see from the graphs, while it is true that width is a parameter that the model can handle when it has little depth, the opposite is not true: in fact, when depth is increased, we observe a significant deterioration in both loss and accuracy.

|  | loss | accuracy| 
|--|--|--|
| training |![train_loss]()   | ![train_acc]()
| validation |![val_loss]()   | ![val_acc]()

## Experiment 1.2
### Parameters
To run this experiment use:

    python main.py --experiment 1_2

For these experiment i use this configuration, that can be found in `/configs/config_exp1_2.yaml`:

   

    #Esperimento 1.2: MLP con e senza connessioni residuali
    seed: 42
    
    data:
      batch_size: 128
      validation_split: 20  #Percentuale che definisce la grandezza del validation set
      num_workers: 4
    
    model:
      type: "MLP"
      params:
        input_size: 784  #28*28 for MNIST
        layers_dim: 128
        class_num: 10
        hidden_layers_num: 2  #Parametro che verrà sovrascritto in esecuzione (prove su profofndità)
        residual: false  #Parametro che verrà sovrascritto in esecuzione (prove su connessioni residuali)
        activation: "ReLU"
        dropout: 0.0
        batch_norm: false
    
    training:
      epochs: 1
      criterion: "CrossEntropyLoss"
      optimizer: "Adam"
      lr: 0.001
      scheduler: "CosineAnnealingLR"
    
    experiment:
      depths: [5, 10, 20, 40]  #Varie profondità di test
      analyze_gradients: true
    
    logging:
      project_name: "LAB1-CNN"
      log_gradients: true

The commented parameters are the ones used for the various runs.
### Results
In the second exercise, we were asked to update our MLP baseline to add the possibility of enabling or disabling residual connections, and then test the model at progressively increasing depths. Analyzing the plots, it is clear that while the absence of residual connections is not a problem at low depths, it becomes an issue as the model’s depth increases.

|  | loss | accuracy| 
|--|--|--|
| training |![train_loss]()   | ![train_acc]()
| validation |![val_loss]()   | ![val_acc]()

This can also be explained by analyzing the gradient plot: it is indeed evident that, without residual connections, the model suffers from the vanishing gradient phenomenon.

|  | loss | accuracy| 
|--|--|--|
| training |![train_loss]()   | ![train_acc]()
| validation |![val_loss]()   | ![val_acc]()
## Experiment 1.3
### Parameters
To run this experiment use:

    python main.py --experiment 1_3

For these experiment i use this configuration, that can be found in `/configs/config_exp1_3.yaml`:

      #Esperimento 1.3: CNN con e senza connessioni residuali su CIFAR-10
    seed: 42
    
    data:
      batch_size: 128
      validation_split: 20  #Percentuale che definisce la grandezza del validation set
      num_workers: 4
    
    model:
      type: "CNN"
      params:
        block_type: "basic"
        layers: [1, 1, 1, 1]  #Parametro che verrà sovrascritto in esecuzione (prove su profofndità)
        num_classes: 10
        residual: false  #Parametro che verrà sovrascritto in esecuzione (prove su connessioni residuali)
        zero_init_residual: true #false: blocco residuale parte inizializzato con valori casuali, true: y simile a x
    
    training:
      epochs: 1
      criterion: "CrossEntropyLoss"
      optimizer: "Adam"
      lr: 0.001
      scheduler: "CosineAnnealingLR"
    
    experiment:
      depths: [[1, 1, 1, 1], [2, 2, 2, 2], [3, 4, 6, 3], [5, 6, 8, 5]]   #Varie profondità di test 
    
    logging:
      project_name: "LAB1-CNN"
      log_gradients: false

The commented parameters are the ones used for the various runs.
### Results

In the third exercise, we were asked to replicate the experiments performed on the MLP, but this time on a CNN using CIFAR10. To do this, I used the implementation of PyTorch’s BasicBlock and ResNet as suggested by the exercise instructions, making it slightly lighter and enabling the choice of whether to use skip connections or not. Analyzing the plots, we can see that at low depths, not only do residual connections not improve performance, but they can even worsen it. The same behavior occurs when the CNN has a very high depth. However, when the CNN has a depth greater than 8 and less than 32, noticeable improvements can be observed.
|  | loss | accuracy| 
|--|--|--|
| training |![train_loss]()   | ![train_acc]()
| validation |![val_loss]()   | ![val_acc]()


## Experiment 2.2
### Parameters
To run this experiment use:

    python main.py --experiment 2_2

For these experiment i use this configuration, that can be found in `/configs/config_exp2_2.yaml`:

    #Esperimento 2.2: Knowledge Distillation
    seed: 0
    
    data:
      batch_size: 128
      validation_split: 20  # percentage
      num_workers: 4
      #Modello teacher (più grande)
    teacher_model:
      type: "CNN"
      params:
        block_type: "basic"
        layers: [3, 4, 6, 3]
        num_classes: 10
        residual: true
        zero_init_residual: true
    
    #Modello studente (più piccolo)
    student_model:
      type: "CNN"
      params:
        block_type: "basic"
        layers: [1, 1, 1, 1]
        num_classes: 10
        residual: false
        zero_init_residual: false
    
    training_teacher:
      epochs: 100
      criterion: "CrossEntropyLoss"
      optimizer: "Adam"
      lr: 0.001
      scheduler: "CosineAnnealingLR"
    
    training_student:
      epochs: 50
      criterion: "CrossEntropyLoss"
      optimizer: "Adam"
      lr: 0.03
      scheduler: "CosineAnnealingLR"
    
    distillation:
      temperature: 3.0
      alpha: 0.7 #0.8 #0.9
    
    logging:
      project_name: "LAB1-CNN"
      log_gradients: false

The commented parameters are the ones used for the various runs.

### Results

For the last exercise, I chose to implement the distillation technique following the guidelines provided in the reference paper [Knowledge Distillation](https://arxiv.org/abs/1503.02531?utm_source=chatgpt.com). The steps are:

-   Teacher Training: train the teacher on CIFAR10 and log its performance.
    
-   Baseline Student: Train a student model on CIFAR10 without teacher logits to serve as a baseline.
    
-   Distillation:
    
    -   Obtain teacher logits for the training set.
        
    -   Create a dataloader combining inputs, true labels, and teacher logits.
        
    -   Train the student using a combined loss:  
        loss = (1−$\alpha$) * hard_loss + $\alpha$ * soft_loss  
        where `hard_loss` = CrossEntropy on true labels and `soft_loss` = KL Divergence with teacher logits.
        
-   Evaluation: Test the distilled student, compare accuracy with teacher and baseline student, and check parameter reduction.

Then I tried to run more experiments by modifying $\alpha$, which is the weight of the losses: as the paper states, increasing $\alpha$ leads to better performance.

![train_loss]() 

The performance improvement of the student was approximately:

 - **2,4%** per $\alpha$ = **0.7**
 - **3,2%** per $\alpha$ = **0.8**
 - **2,8%** per $\alpha$ = **0.9**

The reduction in the number of parameters was approximately **78%**.
