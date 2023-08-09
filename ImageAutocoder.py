import torch

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class PrintShape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x
class CutShape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :, 1:-1, 1:-1]

class ImageAutoencoderLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(128*128, 128*96),
            torch.nn.ReLU(),
            torch.nn.Linear(128*96, 128*64),
            torch.nn.ReLU(),
            torch.nn.Linear(128*64, 128*48),
            torch.nn.ReLU(),
            torch.nn.Linear(128*48, 128*32),
            torch.nn.ReLU(),
            torch.nn.Linear(128*16, 128*8),
            torch.nn.ReLU(),
        )
        self.decoder =  torch.nn.Sequential(
            torch.nn.Linear(128*8, 128*16),
            torch.nn.ReLU(),
            torch.nn.Linear(128*16, 128*32),
            torch.nn.ReLU(),
            torch.nn.Linear(128*32, 128*48),
            torch.nn.ReLU(),
            torch.nn.Linear(128*48, 128*64),
            torch.nn.ReLU(),
            torch.nn.Linear(128*64, 128*96),
            torch.nn.ReLU(),
            torch.nn.Linear(128*96, 128*128),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def code(self, x):
        return self.encoder(x)
    
class ImageAutoencoderConv3R6C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #input should be a grayscale image 128x128, 1x128x128
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (5,5), stride=(1,1), padding=(1,1)), # 32x128x128
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 48, (3,3), stride=(2,2), padding=(1,1)), # 48x64x64
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 64, (3,3), stride=(2,2), padding=(1,1)), # 64x32x32
            torch.nn.Conv2d(64, 80,(3,3), stride=(2,2), padding=(1,1)), #80x16x16
            torch.nn.ReLU(),
            torch.nn.Conv2d(80, 96,(3,3), stride=(2,2), padding=(1,1)), #96x8x8
            torch.nn.Conv2d(96, 128,(3,3), stride=(2,2), padding=(1,1)), #128x4x4
            torch.nn.Flatten(), # Usefull when extracting data from coder
        )
        #
        self.decoder =  torch.nn.Sequential(
            Reshape((-1, 128, 4, 4)), #Because coder flattens the data
            torch.nn.ConvTranspose2d(128, 96, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #96x8x8
            torch.nn.ConvTranspose2d(96, 80, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #80x16x16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(80, 64,(3,3), stride=(2,2), padding=(1,1), output_padding=1), #64x16x16
            torch.nn.ConvTranspose2d(64, 48, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 48x32x32
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(48, 32, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 32x64x64
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, (5,5), stride=(1,1), padding=(1,1)), # 1x128x128
            CutShape(),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def code(self, x):
        return self.encoder(x)

class ImageAutoencoderConv4R6C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #input should be a grayscale image 128x128, 1x128x128
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (5,5), stride=(1,1), padding=(1,1)), # 32x128x128
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 48, (3,3), stride=(2,2), padding=(1,1)), # 48x64x64
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 64, (3,3), stride=(2,2), padding=(1,1)), # 64x32x32
            torch.nn.Conv2d(64, 80,(3,3), stride=(2,2), padding=(1,1)), #80x16x16
            torch.nn.ReLU(),
            torch.nn.Conv2d(80, 96,(3,3), stride=(2,2), padding=(1,1)), #96x8x8
            torch.nn.Conv2d(96, 128,(3,3), stride=(2,2), padding=(1,1)), #128x4x4
            torch.nn.ReLU(),
            torch.nn.Flatten(), # Usefull when extracting data from coder
        )
        #
        self.decoder =  torch.nn.Sequential(
            Reshape((-1, 128, 4, 4)), #Because coder flattens the data
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 96, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #96x8x8
            torch.nn.ConvTranspose2d(96, 80, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #80x16x16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(80, 64,(3,3), stride=(2,2), padding=(1,1), output_padding=1), #64x16x16
            torch.nn.ConvTranspose2d(64, 48, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 48x32x32
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(48, 32, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 32x64x64
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, (5,5), stride=(1,1), padding=(1,1)), # 1x130x130
            CutShape(), #1x128x128
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def code(self, x):
        return self.encoder(x)

class ImageAutoencoderConv0R6C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #input should be a grayscale image 128x128, 1x128x128
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (5,5), stride=(1,1), padding=(1,1)), # 32x128x128
            torch.nn.Conv2d(32, 48, (3,3), stride=(2,2), padding=(1,1)), # 48x64x64
            torch.nn.Conv2d(48, 64, (3,3), stride=(2,2), padding=(1,1)), # 64x32x32
            torch.nn.Conv2d(64, 80,(3,3), stride=(2,2), padding=(1,1)), #80x16x16
            torch.nn.Conv2d(80, 96,(3,3), stride=(2,2), padding=(1,1)), #96x8x8
            torch.nn.Conv2d(96, 128,(3,3), stride=(2,2), padding=(1,1)), #128x4x4
            torch.nn.Flatten(), # Usefull when extracting data from coder
        )
        #
        self.decoder =  torch.nn.Sequential(
            Reshape((-1, 128, 4, 4)), #Because coder flattens the data
            torch.nn.ConvTranspose2d(128, 96, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #96x8x8
            torch.nn.ConvTranspose2d(96, 80, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #80x16x16
            torch.nn.ConvTranspose2d(80, 64,(3,3), stride=(2,2), padding=(1,1), output_padding=1), #64x16x16
            torch.nn.ConvTranspose2d(64, 48, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 48x32x32
            torch.nn.ConvTranspose2d(48, 32, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 32x64x64
            torch.nn.ConvTranspose2d(32, 1, (5,5), stride=(1,1), padding=(1,1)), # 1x128x128
            CutShape(),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def code(self, x):
        return self.encoder(x)

class ImageAutoencoderConv0R5C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #input should be a grayscale image 128x128, 1x128x128
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (3,3), stride=(2,2), padding=(1,1)), # 32x64x64
            torch.nn.Conv2d(32, 64, (3,3), stride=(2,2), padding=(1,1)), # 64x32x32
            torch.nn.Conv2d(64, 80, (3,3), stride=(2,2), padding=(1,1)), # 80x16x16
            torch.nn.Conv2d(80, 96,(3,3), stride=(2,2), padding=(1,1)), #96x8x8
            torch.nn.Conv2d(96, 128,(3,3), stride=(2,2), padding=(1,1)), #128x4x4
            torch.nn.Flatten(), # Usefull when extracting data from coder
        )
        #
        self.decoder =  torch.nn.Sequential(
            Reshape((-1, 128, 4, 4)), #Because coder flattens the data
            torch.nn.ConvTranspose2d(128, 96, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #96x8x8
            torch.nn.ConvTranspose2d(96, 80, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #80x16x16
            torch.nn.ConvTranspose2d(80, 64,(3,3), stride=(2,2), padding=(1,1), output_padding=1), #64x32x32
            torch.nn.ConvTranspose2d(64, 32, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 48x64x64
            torch.nn.ConvTranspose2d(32, 1, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 1x128x128
            CutShape(),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def code(self, x):
        return self.encoder(x)
    
class ImageAutoencoderConv4R5C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #input should be a grayscale image 128x128, 1x128x128
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (3,3), stride=(2,2), padding=(1,1)), # 32x64x64
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (3,3), stride=(2,2), padding=(1,1)), # 64x32x32
            torch.nn.Conv2d(64, 80, (3,3), stride=(2,2), padding=(1,1)), # 80x16x16
            torch.nn.ReLU(),
            torch.nn.Conv2d(80, 96,(3,3), stride=(2,2), padding=(1,1)), #96x8x8
            torch.nn.ReLU(),
            torch.nn.Conv2d(96, 128,(3,3), stride=(2,2), padding=(1,1)), #128x4x4
            torch.nn.ReLU(),
            torch.nn.Flatten(), # Usefull when extracting data from coder
        )
        #
        self.decoder =  torch.nn.Sequential(
            Reshape((-1, 128, 4, 4)), #Because coder flattens the data
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 96, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #96x8x8
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(96, 80, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #80x16x16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(80, 64,(3,3), stride=(2,2), padding=(1,1), output_padding=1), #64x32x32
            torch.nn.ConvTranspose2d(64, 32, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 48x64x64
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 1x128x128
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def code(self, x):
        return self.encoder(x)

class ImageAutoencoderConvColor4R5C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #input should be a color image 128x128, 3x128x128
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, (3,3), stride=(2,2), padding=(1,1)), # 32x64x64
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (3,3), stride=(2,2), padding=(1,1)), # 64x32x32
            torch.nn.Conv2d(64, 80, (3,3), stride=(2,2), padding=(1,1)), # 80x16x16
            torch.nn.ReLU(),
            torch.nn.Conv2d(80, 96,(3,3), stride=(2,2), padding=(1,1)), #96x8x8
            torch.nn.ReLU(),
            torch.nn.Conv2d(96, 128,(3,3), stride=(2,2), padding=(1,1)), #128x4x4
            torch.nn.ReLU(),
            torch.nn.Flatten(), # Usefull when extracting data from coder
        )
        #
        self.decoder =  torch.nn.Sequential(
            Reshape((-1, 128, 4, 4)), #Because coder flattens the data
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 96, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #96x8x8
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(96, 80, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #80x16x16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(80, 64,(3,3), stride=(2,2), padding=(1,1), output_padding=1), #64x32x32
            torch.nn.ConvTranspose2d(64, 32, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 48x64x64
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 3x128x128
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def code(self, x):
        return self.encoder(x)
    
class ImageAutoencoderConvColor3R4C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #input should be a color image 128x128, 3x128x128
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, (3,3), stride=(2,2), padding=(1,1)), # 32x64x64
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (3,3), stride=(2,2), padding=(1,1)), # 64x32x32
            torch.nn.Conv2d(64, 96, (3,3), stride=(4,4), padding=(1,1)), # 96x16x16
            torch.nn.ReLU(),
            torch.nn.Conv2d(96, 128,(3,3), stride=(2,2), padding=(1,1)), #128x4x4
            torch.nn.ReLU(),
            torch.nn.Flatten(), # Usefull when extracting data from coder
        )
        #
        self.decoder =  torch.nn.Sequential(
            Reshape((-1, 128, 4, 4)), #Because coder flattens the data
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 96, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #96x8x8
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(96, 64, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #80x16x16
            torch.nn.ConvTranspose2d(64, 32, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 48x64x64
            torch.nn.ConvTranspose2d(32, 32, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 48x64x64
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 3x128x128
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        x = x[:3,:,:]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def code(self, x):
        return self.encoder(x)
    
class ImageAutoencoderConvColor4R6C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #input should be a color image 128x128, 3x128x128
        self.encoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(3, 48, (3,3), stride=(2,2), padding=(1,1)), # 48x64x64
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 64, (3,3), stride=(2,2), padding=(1,1)), # 64x32x32
            torch.nn.Conv2d(64, 80, (3,3), stride=(2,2), padding=(1,1)), # 80x16x16
            torch.nn.ReLU(),
            torch.nn.Conv2d(80, 96,(3,3), stride=(2,2), padding=(1,1)), #96x8x8
            torch.nn.ReLU(),
            torch.nn.Conv2d(96, 128,(3,3), stride=(2,2), padding=(1,1)), #128x4x4
            torch.nn.ReLU(),
            torch.nn.Flatten(), # Usefull when extracting data from coder
        )
        #
        self.decoder =  torch.nn.Sequential(
            Reshape((-1, 128, 4, 4)), #Because coder flattens the data
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 96, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #96x8x8
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(96, 80, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #80x16x16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(80, 64,(3,3), stride=(2,2), padding=(1,1), output_padding=1), #64x32x32
            torch.nn.ConvTranspose2d(64, 48, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 48x64x64
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(48, 3, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 1x128x128
            torch.nn.ReLU(),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def code(self, x):
        return self.encoder(x)
    
class ImageAutoencoderConvColor6R7BN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #input should be a color image 128x128, 3x128x128
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, (3,3), stride=(2,2), padding=(1,1)), # 32x64x64
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, (3,3), stride=(2,2), padding=(1,1)), # 64x32x32
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 80, (3,3), stride=(2,2), padding=(1,1)), # 80x16x16
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(80),
            torch.nn.Conv2d(80, 96,(3,3), stride=(2,2), padding=(1,1)), #96x8x8
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(96),
            torch.nn.Conv2d(96, 128,(3,3), stride=(2,2), padding=(1,1)), #128x4x4
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 512,(3,3), stride=(2,2), padding=(1,1)), #512x2x2
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(512, 2048,(3,3), stride=(2,2), padding=(1,1)), #2048x1x1
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(2048),
            torch.nn.Flatten(), # Usefull when extracting data from coder
        )
        #
        self.decoder =  torch.nn.Sequential(
            Reshape((-1, 2048, 1, 1)), #Because coder flattens the data
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ConvTranspose2d(2048, 512, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #512x2x2
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.ConvTranspose2d(512, 128, (3,3), stride=(2,2), padding=(1,1), output_padding=1), #128x4x4
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 96,(3,3), stride=(2,2), padding=(1,1), output_padding=1), #96x8x8
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(96),
            torch.nn.ConvTranspose2d(96, 80, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 96x16x16
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(80),
            torch.nn.ConvTranspose2d(80, 64, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 64x32x32
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(64, 32, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 32x64x64
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(32, 3, (3,3), stride=(2,2), padding=(1,1), output_padding=1), # 3x128x128
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3),
            torch.nn.Sigmoid(),
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def code(self, x):
        return self.encoder(x)
    
class EncoderLayer(torch.nn.Module):
    def __init__(self, paramsIn : int, paramsOut : int, downsample = True):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=paramsIn, out_channels=paramsIn, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.BatchNorm2d(num_features=paramsIn),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=paramsIn, out_channels=paramsOut, kernel_size=(3,3), stride=((2,2) if downsample else (1,1)), padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=paramsOut)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=paramsIn, out_channels=paramsOut, kernel_size=(3,3), stride=((2,2) if downsample else (1,1)), padding=(1,1)),
            torch.nn.AvgPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1))
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=paramsOut),
            torch.nn.ReLU()
        )

    def forward(self, x):
        ly1r = self.layer1(x)
        intermr = self.layer2(ly1r)
        intermr += self.layer3(ly1r)        
        intermr = self.layer4(intermr)
        return intermr
class DecoderLayer(torch.nn.Module):
    def __init__(self, paramsIn : int, paramsOut: int, upsample = True):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=paramsIn),
            torch.nn.ConvTranspose2d(in_channels=paramsIn, out_channels=paramsIn, kernel_size=(3,3), stride=((2,2)if upsample else(1,1)), output_padding=((1,1)if upsample else(0,0))),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=paramsIn, out_channels=paramsOut, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=paramsOut)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=paramsIn, out_channels=paramsOut, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=paramsOut, out_channels=paramsOut, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=paramsOut)
        )
    def forward(self, x):
        ly1r = self.layer1(x)[:,:,1:-1,1:-1]
        intermr = self.layer2(ly1r)
        intermr += self.layer3(ly1r)
        intermr = self.layer4(intermr)
        return intermr

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            EncoderLayer(paramsIn=3, paramsOut=32, downsample=True), # 3x128x128 -> 32x64x64
            EncoderLayer(paramsIn=32, paramsOut=64, downsample=True), # 32x64x64 -> 64x32x32
            EncoderLayer(paramsIn=64, paramsOut=128, downsample=True), # 64x32x32 -> 128x16x16
            EncoderLayer(paramsIn=128, paramsOut=256, downsample=True), # 128x16x16 -> 256x8x8
            EncoderLayer(paramsIn=256, paramsOut=512, downsample=True), # 256x8x8-> 512x4x4
            EncoderLayer(paramsIn=512, paramsOut=1024, downsample=True), # 512x4x4 -> 1024x2x2
            EncoderLayer(paramsIn=1024, paramsOut=2048, downsample=True), # 1024x2x2 -> 2048x1x1
        )
        self.decoder = torch.nn.Sequential(
            DecoderLayer(paramsIn=2048, paramsOut= 1024, upsample= True), # 2048x1x1 -> 1024x2x2
            DecoderLayer(paramsIn=1024, paramsOut= 512, upsample= True), # 1024x2x2 -> 512x4x4
            DecoderLayer(paramsIn=512, paramsOut= 256, upsample= True), # 512x4x4 -> 256x8x8
            DecoderLayer(paramsIn=256, paramsOut= 128, upsample= True), # 256x8x8 -> 128x16x16
            DecoderLayer(paramsIn=128, paramsOut= 64, upsample= True), # 128x16x16 -> 64x32x32
            DecoderLayer(paramsIn=64, paramsOut= 32, upsample= True), # 64x32x32 -> 32x64x64
            DecoderLayer(paramsIn=32, paramsOut= 3, upsample= True), # 32x64x64 -> 3x128x128
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded