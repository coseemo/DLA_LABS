from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

#Classe ripresa dal laboratorio 1 per la gestione di CIFAR 10
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
                    (0.4914, 0.4822, 0.4465),   
                    (0.2470, 0.2435, 0.2616),  
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

    #Leggermente modificata ponendo gli shuffle a false per mantenere
    #la corrispondenza in vista dell'utilizzo di y_pred e y_true
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

#Classe per la gestione del dataset FakeData
class FakeDataLoader:
    
    def __init__(self, batch_size, image_size=(3, 32, 32), num_samples=1000, num_workers=4):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_samples = num_samples

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                )
            ]
        )

        #Utilizziamo torchvision.datasets.FakeData
        self.dataset = datasets.FakeData(
            size=self.num_samples,
            image_size=self.image_size,
            transform=self.transform,
        )

    def get_dataloader(self):
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return loader
