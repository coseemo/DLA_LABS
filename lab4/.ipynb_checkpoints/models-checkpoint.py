import torch
import torch.nn as nn
import torch.nn.functional as F

#Modelli ripresi dal notebook del laboratorio, più una cnn discriminatore per l'esercizio su JARN

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = "classifier"
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = "classifier"
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        #Ci interessa solo perturbato/non-perturbato
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_plus(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = "classifier"
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  #downsample
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  #downsample
        
        self.flatten_dim = 256 * 8 * 8  #assuming input is 32x32
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   #-> 32x32x32
        x = F.relu(self.conv2(x))   #-> 32x32x64
        x = F.relu(self.conv3(x))   #-> 32x32x128
        x = F.relu(self.conv4(x))   #-> 16x16x128
        x = F.relu(self.conv5(x))   #-> 8x8x256
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        #Input size: [batch, 3, 32, 32]
        #Output size: [batch, 3, 32, 32]
        self.type = "Autoencoder"
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            #[batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           #[batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           #[batch, 48, 4, 4]
            nn.ReLU(),
#			nn.Conv2d(48, 96, 4, stride=2, padding=1),           #[batch, 96, 2, 2]
#            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  #[batch, 48, 4, 4]
#            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  #[batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  #[batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   #[batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded