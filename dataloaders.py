from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


#Classe che gestisce il dataset MNIST
class MNIST_DataLoader:
    def __init__(self, batch_size, split, num_workers=4):
        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers

        #Si applicano le trasformazioni standard per MNIST
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                ),
            ]
        )

        #Scarichiamo l'intero dataset di training
        full_train_dataset = datasets.MNIST(
            root="./data", train=True, transform=self.transform, download=True
        )

        #Lo dividiamo tra training e validation, utilizzando una percentuale intera (es. 70-30)
        val_size = int(self.split * len(full_train_dataset)/100)
        train_size = len(full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

        #Carichiamo il dataset di test
        self.test_dataset = datasets.MNIST(
            root="./data", train=False, transform=self.transform, download=True
        )

    def get_dataloaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader


#Applichiamo la stessa procedura per CIFAR10 che tuttavia, per affrontare l'esercizio 2.2,
#dovremo dotare della capacità di gestire le soft label ottenute dal modello insegnante
class CIFAR10_DataLoader:
    
    def __init__(self, batch_size, split, num_workers=4):
    
        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers

        #Utilizziamo i valori di normalizzazione per CIFAR10
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),  # Mean for each channel
                    (0.2470, 0.2435, 0.2616),  # Std for each channel
                ),
            ]
        )

        full_train_dataset = datasets.CIFAR10(
            root="./data", train=True, transform=self.transform, download=True
        )

        val_size = int(self.split * len(full_train_dataset)/100)
        train_size = len(full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

        self.test_dataset = datasets.CIFAR10(
            root="./data", train=False, transform=self.transform, download=True
        )

    #Metodo per ottenere i dataloader, la possibilità di non fare shuffle
    #sarà utile per la creazione del dataset di distillazione
    def get_dataloaders(self, shuffle=True):
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader


    #Crea un dataloader per la distillazione usando i dati originali e i logits del teacher.
    def create_distillation_dataloader(self, original_data, teacher_logits):
        
        distillation_dataset = DistillationDataset(original_data, teacher_logits)
        
        distillation_loader = DataLoader(
            distillation_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        
        return distillation_loader
        
        
#Override del dataset per la distillazione
class Distillation_Dataset(Dataset):
    
    def __init__(self, base_dataset, teacher_logits):
        self.base_dataset = base_dataset
        self.teacher_logits = teacher_logits

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        teacher_logit = self.teacher_logits[index]
        return x, y, teacher_logit

    def __len__(self):
        return len(self.base_dataset)
