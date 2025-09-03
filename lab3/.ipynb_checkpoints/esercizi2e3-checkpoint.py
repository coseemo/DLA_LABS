import wandb
from datasets import load_dataset
from logger import Logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def tokenizer_function(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True)

def tokenize_ds(data, tokenizer):
    #Usato la funzione lambda per non hardcodare il nome del modello
    #(preso spunto dalla coumentazione di Dataset.map)
    tokenized_data = data.map(lambda examples: tokenizer_function(examples["text"], tokenizer), batched = True) 
    return tokenized_data

def verify_keys(tk_ds):

    #Lista delle chiavi che devono essere presenti per ogni elemento
    expected_keys = ["text", "label", "input_ids", "attention_mask"]
    message = "The keys are alright"

    print(f"\n{tk_ds[0]}")

    #Funzione che mi dice se tutti gli elementi hanno le chiavi che
    #devono avere. 
    #(magari si poteva fare anche in modo che venissero restituite informazioni anche su quali chiavi
    #(e in quali elementi) fossero mancanti)
    for example in tk_ds:
        for k in expected_keys:
            if(k not in example):
                message = "The keys aren't alright"

    print(message)

def compute_metrics(eval_preds):
    
    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    predicted = logits.argmax(axis=1)

    #Accuracy del modello: corrette/totale
    accuracy = accuracy_score(labels, predicted)
    #Quanti positivi erano veramente positivi: veri postivi / veri positivi+falsi positivi
    precision = precision_score(labels, predicted)
    #Di tutti i positivi quanti ne ha trovati: veri positivi / veri positivi+falsi negativi
    recall = recall_score(labels, predicted)
    #Media armonica tra precision e recall: 2*p*r/(p+r)
    f1 = f1_score(labels, predicted)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    

def esercizi2e3(config):

    ds_name = config['dataset_name']
    model_name = config['model_name']
    ds = load_dataset(ds_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #Esercizio 2.1
    #Tokenizziamo il dataset
    tokenized_train = tokenize_ds(ds["train"], tokenizer)
    tokenized_val = tokenize_ds(ds["validation"], tokenizer)
    tokenized_test = tokenize_ds(ds["test"], tokenizer)

    #Verifichiamo che siano tokenizzati correttamente
    verify_keys(tokenized_train)
    verify_keys(tokenized_val)
    verify_keys(tokenized_test)

    print(ds["train"][0])
    print(set(ds["train"]["label"]))

    #Esercizio 2.2
    #Istanziamo il modello per connettere automaticamente
    #i CLS token dell'ultimo hidden layer al classificatore
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    #Esercizio 2.3
    #Istanziamo il collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    wandb_logger = Logger(
        project_name=config.get("wandb_project", "LAB3-Transformers"),
        run_name=config.get("run_name", None),
        config=config
    )

    #Training arguments
    t_args_cfg = config.get("training_args", {})
    t_args = TrainingArguments(
        output_dir="./results",
        run_name=config.get("run_name", "ex2"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=t_args_cfg.get("learning_rate", 5e-5),
        per_device_train_batch_size=t_args_cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=t_args_cfg.get("per_device_eval_batch_size", 8),
        num_train_epochs=t_args_cfg.get("num_train_epochs", 3),
        weight_decay=t_args_cfg.get("weight_decay", 0.01),
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        #Simula batch più grande
        gradient_accumulation_steps=t_args_cfg.get("gradient_accumulation_steps", 2),
        #Disabilita opzione di conversione da 32 a 16 bit (sto usando la cpu)
        fp16=t_args_cfg.get("fp16", False),
        report_to="wandb",
    )

    #Esercizio 3.1
    #Se lora è abilitato gran parte del modello viene congelata 
    #e andiamo a trainare soltanto alcuni moduli del modello
    if config["lora_args"]["lora"]:
        lora_config = LoraConfig(
            r=config["lora_args"]["rank"],  #rango delle matrici a bassa dimensionalità A e B
            lora_alpha=config["lora_args"]["alpha"],  #fattore di scala che determina l'impatto di A e B sui parametri del modello
            #Moduli del modello su cui andiamo ad applicare lora:
            #Multi-Head Self Attention [q,k,v,out]
            #Feed-Forward [lin1 e lin2]
            #Senza modificare tutto il modelli, modifichiamo ciò a cui "prestiamo attenzione"
            #e come lo valutiamo
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"], 
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        #Inizializzo il modello con lora utilizzando la configurazione sopra
        model = get_peft_model(model, lora_config)

        #Contiamo i parametri totali e quelli apprendibili
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")

    #Configurazione del trainer
    trainer = Trainer(
        model=model,
        args=t_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    #Facciamo il fine-tuning del modello
    trainer.train()

    #Testiamo    
    test_results = trainer.evaluate(eval_dataset=tokenized_test)
    print(test_results)

    wandb.log({
                "test/accuracy": test_results["eval_accuracy"],
                "test/precision": test_results["eval_precision"],
                "test/recall": test_results["eval_recall"],
                "test/f1": test_results["eval_f1"]
            })

    wandb_logger.finish()
