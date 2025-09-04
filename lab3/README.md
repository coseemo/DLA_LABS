# Laboratory 3: Transformers, Fine-Tuning, Lora


## Plots
All the plots can be found here:
- **lab3:** [https://wandb.ai/cosimo-borghini1-universit-di-firenze/LAB3-Transformes?nw=nwusercosimoborghini1]

[If you expand the runs, you can see which parameters i used for each run]

## Experiment 2 and 3.1
### Parameters
To run this experiment use:

    python main.py --experiment 2e3

For these experiment I use this configuration, that can be found in `/configs/config_exp2e3.yaml`:

 

    model_name: distilbert-base-uncased
    dataset_name: cornell-movie-review-data/rotten_tomatoes
    wandb_project: LAB3-Transformers
    run_name: distilbert-ex2
    lora_args: 
      lora: True                         #se lora è attivo oppure no True, False
      rank: 16                            #rango delle matrici        8, 16
      alpha: 64                           #peso delle matrici        32, 64
    training_args:
      learning_rate: 0.00005
      per_device_train_batch_size: 8
      per_device_eval_batch_size: 8
      num_train_epochs: 3
      weight_decay: 0.01
      gradient_accumulation_steps: 2
      fp16: false


The commented parameters are the ones used for the various runs.
### Results
Looking at the training graphs, I notice that with LoRA the training loss remains higher compared to the model without it, yet the validation accuracy is better. This phenomenon could occur because LoRA constrains the adaptation of the main model weights, so the model doesn’t fully minimize the training loss but ends up generalizing better. I also observe that precision and F1 improve with LoRA, while recall decreases. This could happen because the model becomes more conservative in predicting positive classes, reducing false positives but missing some true positives. Overall, this suggests that LoRA acts as an implicit regularizer, enhancing generalization and prediction quality, even if it comes at the cost of slightly lower recall.

| alpha\rank |  8|16|baseline|
|--|--|--|--|
| **32** | **83,6%** |**83,9%** |**81,4%**
|**64** |  **83,7%** | **84,1%**|**81,4%**

| ![recall](https://github.com/coseemo/DLA_LABS/blob/main/lab3/plots3/recall.png) | ![f1](https://github.com/coseemo/DLA_LABS/blob/main/lab3/plots3/f1.png) |
|--|--|
| ![precision](https://github.com/coseemo/DLA_LABS/blob/main/lab3/plots3/precision.png) | ![loss](https://github.com/coseemo/DLA_LABS/blob/main/lab3/plots3/loss.png) |


## Experiment 3.3

### Idea
For exercise 3.3, I wanted to try using DistilGPT2 to attempt to solve the word chains from a TV show called _"Reazione a Catena."_ The game presents contestants with chains like 'cane -> l -> ? -> ?' and asks them to guess the next word based only on the initial letter. Once the word is guessed, the next initial is given to the contestants: 'cane -> lupo -> b -> ?'. [The solution for the proposed chain is 'cane -> lupo -> bianco -> latte.']

![reazione a catena](https://github.com/coseemo/DLA_LABS/blob/main/lab3/plots3/reazione%20a%20catena.jpg)

I first tried to build a dataset of word connections: I downloaded the collocations dictionary from [this link](https://downloads.freemdict.com/%E5%B0%9A%E6%9C%AA%E6%95%B4%E7%90%86/%E5%85%B1%E4%BA%AB2020.5.11/content/4_others/italian/Dizionario%20delle%20collocazioni%20Le%20combinazioni%20delle%20parole%20in%20italiano/?utm_source=chatgpt.com) and used this tool to convert it from .mdx to .json: [pyglossary](https://github.com/ilius/pyglossary/tree/master?utm_source=chatgpt.com)

After that, I tried to automatically clean the dictionary entries by filtering out articles and dictionary-specific nomenclature

    Prime 10 parole del dizionario pulito:
    1. abbagliare
    2. abbaglio
    3. abbaiare
    4. abbandonare
    5. abbandonarsi
    6. abbandono
    7. abbassamento
    8. abbassare
    9. abbattere
    10. abbigliamento

I then built a "bidirectional" dictionary so that each connection could also serve as a key

    Prime 10 parole del dizionario:
    1. abbagliare
    2. offuscare
    3. illudere
    4. vista
    5. abbaglio
    6. fatale
    7. totale
    8. lieve
    9. solito
    10. tremendo

 and created a dataset of 50,000 entries composed of examples like: 

> cane -> l [ANSWER] lupo

 and 

> 'lupo -> c [ANSWER] cane'

I also try to creating some chains like these:

    Catena 1: cuocere -> bene -> prezioso -> documento -> reperire -> materiale
    Catena 2: case -> gruppo -> pressione -> sentire -> vocazione -> scoprire
    Catena 3: medicina -> tollerare -> sopraffazione -> atto -> crudele -> sovrano
    Catena 4: comunale -> imposta -> pagare -> cifra -> denaro -> usare
    Catena 5: diocesano -> palazzo -> popolare -> canzone -> d'autore -> quadro

Then, I finetuned the model

### Results

The result is not excellent; the words are often "unexpected" and sometimes are not even Italian words. In my opinion, the reasons for this "failure" are: the difficulty of automatically—or manually—creating a dataset that must be based on phrases, proverbs, and idiomatic expressions, and also resource limitations, since I conducted this experiment in the middle of summer without access to a GPU.

Nevertheless, if you want to have some "fun," you can go to the end of the `ChainReaction` notebook. By running the last cell, you can provide a word and an initial to the model and see the top 5 predictions it generates.

