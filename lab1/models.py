import torch
import torch.nn as nn
import torch.optim as optim


#Definizione della classe MLP_Block sfruttando 
#l'ovveride di nn.Module (utile per la gestione dei parametri e modalità).
#Tale classe costituisce i moduli che andranno a formare l'MLP.
class MLP_Block(nn.Module):
    
    def __init__(self, in_size, out_size, activation="ReLU", dropout=0.0, batch_norm=False):
        super().__init__()

        #Tramite getattr gestisco l'utilizzo di possibili attivazioni diverse
        self.activation = getattr(nn, activation)

        #Ogni blocco è un insieme di layer sequenziali:
        if batch_norm:
            self.block = nn.Sequential(
                nn.Linear(in_size, out_size),          #layer completamente connesso 
                nn.BatchNorm1d(out_size),              #applicazione della normalizzazione
                self.activation(),                     #applicazione dell'attivazione 
                nn.Dropout(dropout)                    #applicazione del dropout
            )

        #Se batch_norm=False, non normalizza
        else:
            self.block = nn.Sequential(
                nn.Linear(in_size, out_size),
                self.activation(),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.block(x)

#Definizione della classe MLP che sfrutta MLP_Block (sempre con override di nn.Module) 
#e con possibilità di utilizzare delle connessioni residuali
class MLP(nn.Module):
    
    def __init__(self, input_size, layers_dim, class_num, hidden_layers_num, residual=False, 
                 activation="ReLU", dropout=0.0, batch_norm=False):
        super().__init__()

        #Flag che gestisce le connessioni residuali
        self.residual = residual

        #Blocco che gestisce i dati in input
        self.input_layer =  MLP_Block(input_size, layers_dim, activation, dropout, batch_norm)

        #Si utilizza ModuleList, per facilitare l'iterazione attraverso i layer nascosti
        self.hidden_layers = nn.ModuleList()

        #Si utilizza una lista di MLP_Block per modellare gli strati nascosti
        for _ in range(hidden_layers_num):
            self.hidden_layers.append(MLP_Block(layers_dim, layers_dim, activation, dropout, batch_norm))

        #Infine si ha un layer completamente connesso che gestisce l'output della rete
        self.output_layer = nn.Linear(layers_dim, class_num)

    #Override del metodo Forward di nn.Module
    def forward(self, x):

        #I dati in input vengono trasformati in un tensore monodimensionale
        x = torch.flatten(x, start_dim=1) 

        #Il tensore viene elaborato dal layer di input
        x = self.input_layer(x)

        #Viene poi elaborato dagli strati nascosti della rete,
        #utilizzando le connessioni residue se il flag è attivo
        if self.residual:
            for layer in self.hidden_layers:
                res = x
                x = layer(x)
                x += res
        else:
            for layer in self.hidden_layers:
                x = layer(x)

        #Infine si ottiene l'output della rete
        out = self.output_layer(x)

        #e lo si ritorna.
        return out


#Definizione della classe CNN_Block prendendo spunto dall'implementazione di BasicBlock
#l'ovveride di nn.Module e la definizione del blocco proposta nel modello resnet(.py) di torchvision.
#Tale classe costituisce i moduli che andranno a formare l'MLP.
class CNN_Block(nn.Module):

    #Serve a regolare il numero di canali in uscita: 
    #in questo caso i numero di canali in uscita resta invariato
    #(nel modulo Bottleneck in resnet.py, troviamo infatti expansion: int = 4)
    expansion: int = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, norm_layer=None, residual=False):
        super().__init__()

        self.norm_layer = nn.BatchNorm2d   

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False)     #Prima convoluzione
        self.bn1 = self.norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)        #Seconda convoluzione
        self.bn2 = self.norm_layer(out_planes)
        self.downsample = downsample                                                                #Utile nei layer più profondi delle ResNet:
                                                                                                    #aiuta a mantenere le dimensioni comparabili
        self.stride = stride
        self.residual = residual                                                                    #Flag per le connessioni residuali

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        out = self.relu(out)
        return out

class CNN(nn.Module):

    #Si costruisce la rete secondo i parametri riportati in "Deep Residual Learning for Image Recognition"
    #Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, CVPR 2016.
    
    def __init__(
        self,
        block_type="basic",
        layers=[2, 2, 2, 2],
        num_classes=10,
        residual=True,
        zero_init_residual=False,
    ):
        super(CNN, self).__init__()

        self.inplanes = 16
        self.residual = residual

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)    #Convoluzione iniziale
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Selezione del tipo di blocco
        if block_type == "basic":
            block = CNN_Block
            
        elif block_type == "bottleneck":
            block = Bottleneck
        
        else:
            raise ValueError("must be basic or bottleneck")

        #Si seguono le specifiche utilizzate nel paper di riferimento per gli esperimenti su CIFAR-10
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        #Se cambia la dimensione o il numero di canali, serve un downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.residual:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        
        #Primo blocco con eventuale downsampling
        layers.append(block(self.inplanes, planes, stride, downsample, residual=self.residual))
        self.inplanes = planes * block.expansion
        
        #Blocchi successivi
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=self.residual))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
