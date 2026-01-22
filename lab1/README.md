
# Laboratory 1: MLP, CNN, Residual Conections and Distillation

## Organization
Most of the code resides in two files: **main.py**, which contains the experiment functions, and **runners.py**, which contains the majority of the program’s logic.


## Plots
All the plots can be found here:
- **lab1:** [https://wandb.ai/cosimo-borghini1-universit-di-firenze/LAB1-CNN?nw=nwusercosimoborghini1]

## Experiment 1.1
### Parameters
To run this experiment use:

    python main.py --experiment 1_1

For these experiment I use this configuration, that can be found in `/configs/config_exp1_1.yaml`:

 

    #Esperimento 1.1: MLP con MNIST
    seed: 42
    
    data:
      batch_size: 128
      #Percentuale che definisce la grandezza del validation set
      validation_split: 20  
      num_workers: 4
    
    model:
      type: "MLP"
      params:
        input_size: 784 #28*28 per MNIST
        #come dimensione dei layer ho fatto prove su [32, 64, 128]
        layers_dim: 128 
        class_num: 10
        #come numero di hidden layers ho fatto prove su [10, 20, 40]
        hidden_layers_num: 40 
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
| training |![train_loss](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/1/W%26B%20Chart%2003_09_2025%2C%2012_50_07.png)   | ![train_acc](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/1/W%26B%20Chart%2003_09_2025%2C%2012_50_45.png)
| validation |![val_loss](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/1/W%26B%20Chart%2003_09_2025%2C%2012_50_54.png)   | ![val_acc](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/1/W%26B%20Chart%2003_09_2025%2C%2012_51_03.png)

## Experiment 1.2
### Parameters
To run this experiment use:

    python main.py --experiment 1_2

For these experiment i use this configuration, that can be found in `/configs/config_exp1_2.yaml`:

   

    #Esperimento 1.2: MLP con e senza connessioni residuali
    seed: 42
    
    data:
      batch_size: 128

      #Percentuale che definisce la grandezza del validation set
      validation_split: 20  
      num_workers: 4
    
    model:
      type: "MLP"
      params:
        input_size: 784  #28*28 for MNIST
        layers_dim: 128
        class_num: 10

        #Parametro che verrà sovrascritto in esecuzione (prove su profofndità)
        hidden_layers_num: 2

        #Parametro che verrà sovrascritto in esecuzione (prove su connessioni residuali)
        residual: false 
        activation: "ReLU"
        dropout: 0.0
        batch_norm: false
    
    training:
      epochs: 30
      criterion: "CrossEntropyLoss"
      optimizer: "Adam"
      lr: 0.001
      scheduler: "CosineAnnealingLR"
    
    experiment:
      #Varie profondità di test
      depths: [5, 10, 20, 40]  
      analyze_gradients: true
    
    logging:
      project_name: "LAB1-CNN"
      log_gradients: true

The commented parameters are the ones used for the various runs.
### Results
In the second exercise, we were asked to update our MLP baseline to add the possibility of enabling or disabling residual connections, and then test the model at progressively increasing depths. Analyzing the plots, it is clear that while the absence of residual connections is not a problem at low depths, it becomes an issue as the model’s depth increases.

|  | loss | accuracy| 
|--|--|--|
| test |![train_loss](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/2/tl.png)   | ![train_acc](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/2/ta.png)

|  | nores | res| 
|--|--|--|
| depth 10 |![nores10](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/2/10f.png)   | ![res10](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/2/10t.png)
| depth 40 |![nores40](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/2/40f.png)   | ![res40](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/2/40t.png)

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
        
        #Parametro che verrà sovrascritto in esecuzione (prove su profofndità)
        layers: [1, 1, 1, 1] 
        
        num_classes: 10
        
        #Parametro che verrà sovrascritto in esecuzione (prove su connessioni residuali)
        residual: false
        
        #false: blocco residuale parte inizializzato con valori casuali, true: y simile a x
        zero_init_residual: true 
    
    training:
      epochs: 100
      criterion: "CrossEntropyLoss"
      optimizer: "Adam"
      lr: 0.001
      scheduler: "CosineAnnealingLR"
    
    experiment:
    #Varie profondità di test 
      depths: [[1, 1, 1, 1], [2, 2, 2, 2], [3, 4, 6, 3], [5, 6, 8, 5]]   
    
    logging:
      project_name: "LAB1-CNN"
      log_gradients: false

The commented parameters are the ones used for the various runs.
### Results

In the third exercise, we were asked to replicate the experiments performed on the MLP, but this time on a CNN using CIFAR10. To do this, I used the implementation of PyTorch’s BasicBlock and ResNet as suggested by the exercise instructions, making it slightly lighter and enabling the choice of whether to use skip connections or not. Analyzing the plots, we can see that at low depths, not only do residual connections not improve performance, but they can even worsen it. The same behavior occurs when the CNN has a very high depth. However, when the CNN has a depth greater than 8 and less than 32, noticeable improvements can be observed.
|  | loss | accuracy| 
|--|--|--|
| training |![train_loss](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/3/3.png)   | ![train_acc](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/3/4.png)
| validation |![val_loss](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/3/5.png)   | ![val_acc](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/3/6.png)
| test | ![test_loss](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/3/1.png) | ![test_acc](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/3/2.png)


## Experiment 2.2
### Parameters
To run this experiment use:

    python main.py --experiment 2_2

For these experiment i use this configuration, that can be found in `/configs/config_exp2_2.yaml`:

    #Esperimento 2.2: Knowledge Distillation
    seed: 0
    
    data:
      batch_size: 128
      validation_split: 20  
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
      lr: 0.001
      scheduler: "CosineAnnealingLR"
    
    distillation:
      temperature: 3.0
      
      #Ho fatto prove con [0.7, 0.8, 0.9]
      alpha: 0.7 
    
    logging:
      project_name: "LAB1-CNN"
      log_gradients: false

The commented parameters are the ones used for the various runs.

### Results

For the last exercise, I choose to implement the distillation technique following the guidelines provided in the reference paper [Knowledge Distillation](https://arxiv.org/abs/1503.02531). The steps are:

-   Teacher Training: train the teacher on CIFAR10 and log its performance.
    
-   Baseline Student: Train a student model on CIFAR10 without teacher logits to serve as a baseline.
    
-   Distillation:
    
    -   Obtain teacher logits for the training set.
        
    -   Create a dataloader combining inputs, true labels, and teacher logits.
        
    -   Train the student using a combined loss:  
        loss = (1−alpha) * hard_loss + alpha * soft_loss  
        where `hard_loss` = CrossEntropy on true labels and `soft_loss` = KL Divergence with teacher logits.
        
-   Evaluation: Test the distilled student, compare accuracy with teacher and baseline student, and check parameter reduction.

Then I tried to run more experiments by modifying alpha, which is the weight of the losses: alpha=0.8 leads to better performance.

![test_loss](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/4/loss.png) 
![test_acc](https://github.com/coseemo/DLA_LABS/blob/main/lab1/plots1/4/acc.png) 

The performance improvement of the student was approximately:

 - **2,4%** per alpha = **0.7**
 - **3,2%** per alpha = **0.8**
 - **2,8%** per alpha = **0.9**

The reduction in the number of parameters was approximately **78%**.
