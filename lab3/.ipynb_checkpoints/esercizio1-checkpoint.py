import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from datasets import load_dataset, get_dataset_split_names, load_dataset_builder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def explore_dataset(ds, ds_name):

    print("\nThe Dataset:", f"\nName:{ds_name}")

    #Ne osservo la struttura
    print("\nDataset Features:")
    ds_builder = load_dataset_builder(ds_name)
    print(ds_builder.info.features)

    #Guardo gli split
    print("\nThe split structure and example:")
    split = get_dataset_split_names(ds_name)
    print(split)

    #Per ogni split conto il numero di esempi positivi e negativi e ne stampo un esempio
    for split_name, split_data in ds.items():
        
        pos = sum(1 for x in split_data["label"] if x == 1)
        neg = sum(1 for x in split_data["label"] if x == 0)
        print(f"\nSplit: {split_name}, Size: {len(split_data)}, Positive_Size: {pos}, Negative_Size: {neg}") 

        pos_ex = next(x for x in split_data if x["label"] == 1)
        neg_ex = next(x for x in split_data if x["label"] == 0)

        print(f"\nExample:\nPositive: Text: {pos_ex["text"]} Label: {pos_ex["label"]}")
        print(f"\nExample:\nNegative: Text: {neg_ex["text"]} Label: {neg_ex["label"]}")

        #Osservo alcuni valori sulle lunghezze delle recensioni: massimo, minimo, media 
        lens = [len(x.split()) for x in split_data["text"]]
        print(f"\nNumber of words per review:\nAvg: {round(sum(lens)/len(lens))}, Min: {min(lens)}, Max: {max(lens)}")

def analyze_outputs(tokenizer, model, ds_name):

    ds = load_dataset(ds_name)
    samples = []

    for split_name, split_data in ds.items():

        samples.append(next(x["text"] for x in split_data if x["label"] == 1))
        samples.append(next(x["text"] for x in split_data if x["label"] == 0))

    inputs = tokenizer(samples, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    print("\nKeys in model output:\n", outputs.keys())

    #Stampo frase originale, token e embedding
    for i, text in enumerate(samples):
        print(f"\n=== Text {i+1} ===")
        print("\nOriginal Text:\n", text)
    
        #Converto input_ids in token reali
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i])
        print("\nGenerated Tokens:\n", tokens)
        #Notiamo che il modello preferisce utilizzare un token solo per parole
        #brevi, mentre utilizza più token per parole più lunghe, attuando una
        #sorta di divisione sillabica segnalata con un doppio cancelletto.
        #Il numero medio di token sarà perciò maggiore uguale della media
        #delle parole calcolate nel punto 1.1
        #Si notano inoltre i token speciali come [SEP] che individua il 
        #punto fermo all'interno della frase e [PAD] che indica il padding
        #che è stato usato per uniformare la frase alle dimensioni di input
        #richieste dal modello (tutti a destra come da default).
    
        #Embeddings del modello
        embeddings = outputs.last_hidden_state[i]
        print("\nShape embedding:\n", embeddings.shape)
        print("\nFirst Token Embedding [CLS]:\n", embeddings[0][:10])
        print("\nSecond Token Embedding (a random word/syllable):\n", embeddings[1][:10])
        #Andiamo ad analizzare la forma del tensore finale e gli embeddings ottenuti:
        #sia quelli per il CLS token, sia quelli per il token di una parola/sillaba
        #qualsiasi, ciò per vedere se vi sono differenze sostanziali tra le rappresentazioni

def verify(model, tokenizer, model_name):

    #Dall'esercizio precedente si apprende che il primo embedding
    #che si ottiene dal modello è proprio quello del [CLS] token.
    #Verifichiamo dunque che tale embedding sia presente tra 
    #quelli ottenuti tramite pipeline con una frase di esempio.
    text = "The butcher of Blaviken"
    print("Original Text:\n", f"{text}")

    #Manuale
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
            outputs = model(**inputs)
    embeddings_a = outputs.last_hidden_state[0]
    a = embeddings_a[0]
    print("\nFirst Token Embedding [CLS]:\n", a)
 
    #Automatico
    feature_extractor = pipeline("feature-extraction", framework="pt", model=model_name, tokenizer=model_name)
    embeddings_b = feature_extractor(text, return_tensors = "pt")[0]

    print("\nEmbeddings automatic extraction:\n", embeddings_b[:10])

    
    for i in range(len(embeddings_b)-1):
        if(torch.equal(a, embeddings_b[i])):
            index = i

    print(f"\nFound, in index {index}")

def extract(model_name, texts):

    #Dopo aver visto che il class token si trova nella prima posizione
    #dell'array di array restitutitoci da pipeline, possiamo allora
    #passare tutti i testi a pipeline ed estrarre il primo elemento
    #(ovvero la rappresentazione del class token).

    print("Extraction start")

    feature_extractor = pipeline("feature-extraction", framework="pt", model=model_name, tokenizer=model_name)
    x = []

    #Passo ogni testo all'estrattore di feature e aggiungo
    #l'array risultante all'array x
    for text in texts:
        CLS_rep = feature_extractor(text)[0][0]
        x.append(CLS_rep)

    print("Extraction complete")

    #Formatto l'array in versione np che è quella
    #richiesta da SVC
    return np.array(x)


def esercizio1(model_name = "distilbert-base-uncased", ds_name = "cornell-movie-review-data/rotten_tomatoes"):

    #esercizio 1.1
    ds = load_dataset(ds_name)
    explore_dataset(ds, ds_name)

    #esercizio 1.2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("\nThe tokenizer:\n", tokenizer)
    model = AutoModel.from_pretrained(model_name)
    print("\nThe model:\n", model)
    analyze_outputs(tokenizer, model, ds_name)

    #Esercizio 1.3
    verify(model, tokenizer, model_name)

    ds = load_dataset(ds_name)
    x_train = extract(model_name, ds["train"]["text"])
    y_train = np.array( ds["train"]["label"])
    x_val = extract(model_name, ds["validation"]["text"])
    y_val = np.array( ds["validation"]["label"])
    x_test = extract(model_name, ds["test"]["text"])
    y_test = np.array( ds["test"]["label"])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    #Addestriamo il classificatore
    print("\nTraining start")
    #Si allena fino alla convergenza (max_iter = -1)
    clf = SVC(kernel='linear', max_iter = -1) 
    clf.fit(x_train, y_train)
    print("Training complete")

    #Valutiamolo
    print("\nEvaluation start:\n")
    val_preds = clf.predict(x_val)
    test_preds = clf.predict(x_test)
    print("Validation Accuracy:", accuracy_score(y_val, val_preds))
    print("Test Accuracy:", accuracy_score(y_test, test_preds))

    #Validation Accuracy: 0.8095684803001876
    #Test Accuracy: 0.7908067542213884



