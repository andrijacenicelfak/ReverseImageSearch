import time
import torch
import torch.nn as nn
import numpy as np
import cv2

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = True):
        super(EncoderLayer, self).__init__()
        if downsample:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride= 1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        start = x if self.downsample is None else self.downsample(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + start
        return self.relu(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upsample = True):
        super(DecoderLayer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        if upsample:
            self.layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
            )
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        else:
            self.layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, output_padding=0, bias=False)
            )
        
    def forward(self, x):
        start = x if self.upsample is None else self.upsample(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x + start

class AutoEncoderDecoder(nn.Module):
    def __init__(self):
        super(AutoEncoderDecoder, self).__init__()

        self.encoderl1 = EncoderLayer(in_channels=3, out_channels=16, downsample=True)      # 3 x 128x128 -> 16 x 64x64
        self.encoderl2 = EncoderLayer(in_channels=16, out_channels=32, downsample=True)     # 16 x 64x64 -> 32 x 32x32
        self.encoderl3 = EncoderLayer(in_channels=32, out_channels=64, downsample=True)     # 32 x 64x64 -> 64 x 16x16
        self.encoderl4 = EncoderLayer(in_channels=64, out_channels=128, downsample=True)    # 64 x 16x16 -> 128 x 8x8
        self.encoderl5 = EncoderLayer(in_channels=128, out_channels=256, downsample=True)   # 128 x 8x8 -> 256 x 4x4
        self.encoderl6 = EncoderLayer(in_channels=256, out_channels=512, downsample=True)   # 256 x 4x4 -> 512 x 2x2

        self.vectorLenght = 2048

        self.decoderl1 = DecoderLayer(in_channels=512, out_channels=256, upsample= True)
        self.decoderl2 = DecoderLayer(in_channels=256, out_channels=128, upsample= True)
        self.decoderl3 = DecoderLayer(in_channels=128, out_channels=64, upsample= True)
        self.decoderl4 = DecoderLayer(in_channels=64, out_channels=32, upsample= True)
        self.decoderl5 = DecoderLayer(in_channels=32, out_channels=16, upsample= True)
        self.decoderl6 = DecoderLayer(in_channels=16, out_channels=3, upsample= True)
    
    def encode(self, x):
        x = self.encoderl1(x)
        x = self.encoderl2(x)
        x = self.encoderl3(x)
        x = self.encoderl4(x)
        x = self.encoderl5(x)
        x = self.encoderl6(x)
        return x

    def decode(self, x):
        x = self.decoderl1(x)
        x = self.decoderl2(x)
        x = self.decoderl3(x)
        x = self.decoderl4(x)
        x = self.decoderl5(x)
        x = self.decoderl6(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
class AutoEncoderDecoderS(nn.Module):
    def __init__(self):
        super(AutoEncoderDecoderS, self).__init__()

        self.encoderl1 = EncoderLayer(in_channels=3, out_channels=8, downsample=True)      #3 x 128x128 -> 8 x 64x64
        self.encoderl2 = EncoderLayer(in_channels=8, out_channels=16, downsample=True)     # 8 x 64x64 -> 16 x 32x32
        self.encoderl3 = EncoderLayer(in_channels=16, out_channels=32, downsample=True)     # 16 x 64x64 -> 32 x 16x16
        self.encoderl4 = EncoderLayer(in_channels=32, out_channels=64, downsample=True)    # 32 x 16x16 -> 64 x 8x8
        self.encoderl5 = EncoderLayer(in_channels=64, out_channels=96, downsample=True)   # 64 x 8x8 -> 96 x 4x4
        self.encoderl6 = EncoderLayer(in_channels=96, out_channels=128, downsample=True)   # 96 x 4x4 -> 128 x 2x2
        self.encoderl7 = EncoderLayer(in_channels=128, out_channels=256, downsample=True)   # 128 x 2x2 -> 256 x 1x11

        self.vectorLenght = 256

        self.decoderl1 = DecoderLayer(in_channels=256, out_channels=128, upsample= True)
        self.decoderl2 = DecoderLayer(in_channels=128, out_channels=96, upsample= True)
        self.decoderl3 = DecoderLayer(in_channels=96, out_channels=64, upsample= True)
        self.decoderl4 = DecoderLayer(in_channels=64, out_channels=32, upsample= True)
        self.decoderl5 = DecoderLayer(in_channels=32, out_channels=16, upsample= True)
        self.decoderl6 = DecoderLayer(in_channels=16, out_channels=8, upsample= True)
        self.decoderl7 = DecoderLayer(in_channels=8, out_channels=3, upsample= True)
    
    def encode(self, x):
        x = self.encoderl1(x)
        x = self.encoderl2(x)
        x = self.encoderl3(x)
        x = self.encoderl4(x)
        x = self.encoderl5(x)
        x = self.encoderl6(x)
        x = self.encoderl7(x)
        return x

    def decode(self, x):
        x = self.decoderl1(x)
        x = self.decoderl2(x)
        x = self.decoderl3(x)
        x = self.decoderl4(x)
        x = self.decoderl5(x)
        x = self.decoderl6(x)
        x = self.decoderl7(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class AutoEncoderDecoderXS(nn.Module):
    def __init__(self):
        super(AutoEncoderDecoderXS, self).__init__()

        self.encoderl1 = EncoderLayer(in_channels=3, out_channels=8, downsample=True)      #3 x 128x128 -> 8 x 64x64
        self.encoderl2 = EncoderLayer(in_channels=8, out_channels=16, downsample=True)     # 8 x 64x64 -> 16 x 32x32
        self.encoderl3 = EncoderLayer(in_channels=16, out_channels=32, downsample=True)     # 16 x 64x64 -> 32 x 16x16
        self.encoderl4 = EncoderLayer(in_channels=32, out_channels=64, downsample=True)    # 32 x 16x16 -> 64 x 8x8
        self.encoderl5 = EncoderLayer(in_channels=64, out_channels=96, downsample=True)   # 64 x 8x8 -> 96 x 4x4
        self.encoderl6 = EncoderLayer(in_channels=96, out_channels=128, downsample=True)   # 96 x 4x4 -> 128 x 2x2
        self.encoderl7 = EncoderLayer(in_channels=128, out_channels=128, downsample=True)   # 128 x 2x2 -> 128 x 1x1

        self.vectorLenght = 128

        self.decoderl1 = DecoderLayer(in_channels=128, out_channels=128, upsample= True)
        self.decoderl2 = DecoderLayer(in_channels=128, out_channels=96, upsample= True)
        self.decoderl3 = DecoderLayer(in_channels=96, out_channels=64, upsample= True)
        self.decoderl4 = DecoderLayer(in_channels=64, out_channels=32, upsample= True)
        self.decoderl5 = DecoderLayer(in_channels=32, out_channels=16, upsample= True)
        self.decoderl6 = DecoderLayer(in_channels=16, out_channels=8, upsample= True)
        self.decoderl7 = DecoderLayer(in_channels=8, out_channels=3, upsample= True)
    
    def encode(self, x):
        x = self.encoderl1(x)
        x = self.encoderl2(x)
        x = self.encoderl3(x)
        x = self.encoderl4(x)
        x = self.encoderl5(x)
        x = self.encoderl6(x)
        x = self.encoderl7(x)
        return x

    def decode(self, x):
        x = self.decoderl1(x)
        x = self.decoderl2(x)
        x = self.decoderl3(x)
        x = self.decoderl4(x)
        x = self.decoderl5(x)
        x = self.decoderl6(x)
        x = self.decoderl7(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class AutoEncoderDecoderM(nn.Module):
    def __init__(self):
        super(AutoEncoderDecoderM, self).__init__()

        self.encoderl1 = EncoderLayer(in_channels=3, out_channels=16, downsample=False)     # 3 x 128x128 -> 16 x 128x128
        self.encoderl2 = EncoderLayer(in_channels=16, out_channels=16, downsample=True)     # 16 x 128x128 -> 16 x 64x64
        self.encoderl3 = EncoderLayer(in_channels=16, out_channels=32, downsample=True)     # 16 x 64x64 -> 32 x 32x32
        self.encoderl4 = EncoderLayer(in_channels=32, out_channels=64, downsample=False)    # 32 x 32x32 -> 64 x 32x32
        self.encoderl5 = EncoderLayer(in_channels=64, out_channels=64, downsample=True)     # 64 x 32x32 -> 64 x 16x16
        self.encoderl6 = EncoderLayer(in_channels=64, out_channels=128, downsample=True)    # 64 x 16x16 -> 128 x 8x8
        self.encoderl7 = EncoderLayer(in_channels=128, out_channels=256, downsample=True)   # 128 x 8x8 -> 256 x 4x4
        self.encoderl8 = EncoderLayer(in_channels=256, out_channels=512, downsample=False)  # 256 x 4x4 -> 512 x 4x4

        self.vectorLenght = 8192

        self.decoderl1 = DecoderLayer(in_channels=512, out_channels=256, upsample= False)
        self.decoderl2 = DecoderLayer(in_channels=256, out_channels=128, upsample= True)
        self.decoderl3 = DecoderLayer(in_channels=128, out_channels=128, upsample= False)
        self.decoderl4 = DecoderLayer(in_channels=128, out_channels=64, upsample= True)
        self.decoderl5 = DecoderLayer(in_channels=64, out_channels=32, upsample= True)
        self.decoderl6 = DecoderLayer(in_channels=32, out_channels=32, upsample= False)
        self.decoderl7 = DecoderLayer(in_channels=32, out_channels=16, upsample= True)
        self.decoderl8 = DecoderLayer(in_channels=16, out_channels=3, upsample= True)
        
    
    def encode(self, x):
        x = self.encoderl1(x)
        x = self.encoderl2(x)
        x = self.encoderl3(x)
        x = self.encoderl4(x)
        x = self.encoderl5(x)
        x = self.encoderl6(x)
        x = self.encoderl7(x)
        x = self.encoderl8(x)
        return x

    def decode(self, x):
        x = self.decoderl1(x)
        x = self.decoderl2(x)
        x = self.decoderl3(x)
        x = self.decoderl4(x)
        x = self.decoderl5(x)
        x = self.decoderl6(x)
        x = self.decoderl7(x)
        x = self.decoderl8(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

def testModel(model, dataLoader, criterion, numTests = 30, shape=(3, 128, 128)):
    for i, (input, _) in enumerate(dataLoader):
        input = input.cuda(non_blocking= True)
        output = model(input)
        loss = criterion(output, input)
        print(f"Current loss : {loss.item()}")

        input = input.view(shape).cpu().detach().numpy().transpose(1,2,0)
        output = output.view(shape).cpu().detach().numpy().transpose(1,2,0)

        hor = np.concatenate((input, output), axis=1)
        hor = cv2.cvtColor(hor, cv2.COLOR_BGR2RGB)
        cv2.imshow("images", hor)

        while cv2.waitKey(0) != ord(' '):
            time.sleep(0.1)

        if i > numTests:
            break
    cv2.destroyAllWindows()
    return
def testModelpPrintEncoded(model, dataLoader, criterion, numTests = 30, shape=(3, 128, 128)):
    all_encoded = []
    for i, (input, _) in enumerate(dataLoader):
        input = input.cuda(non_blocking= True)
        encoded = model.encode(input)
        output = model.decode(encoded)
        loss = criterion(output, input)
        all_encoded.append(encoded.cpu().detach().numpy().flatten())
        print(f"Current loss : {loss.item()}")
        for e in zip(*all_encoded):
            print(e)
        input = input.view(shape).cpu().detach().numpy().transpose(1,2,0)
        output = output.view(shape).cpu().detach().numpy().transpose(1,2,0)

        hor = np.concatenate((input, output), axis=1)
        hor = cv2.cvtColor(hor, cv2.COLOR_BGR2RGB)
        cv2.imshow("images", hor)

        while cv2.waitKey(0) != ord(' '):
            time.sleep(0.1)

        if i > numTests:
            break
    cv2.destroyAllWindows()
    return

def testModelMultiple(models, dataLoader, numTests = 30, shape=(3, 128, 128), save = False, saveName = ""):
    index = 1

    for i, (input, _) in enumerate(dataLoader):
        input = input.cuda(non_blocking= True)
        outputs = []
        for model in models:
            outputs.append(model(input))

        input = input.view(shape).cpu().detach().numpy().transpose(1,2,0)
        outputs = [output.view(shape).cpu().detach().numpy().transpose(1,2,0) for output in outputs]
        outputs.insert(0, input)
        hor = np.concatenate(outputs, axis=1)
        hor = cv2.cvtColor(hor, cv2.COLOR_BGR2RGB)
        cv2.imshow("images", hor)

        key = cv2.waitKey(0)
        if  key == ord(' ') and save:
            cv2.imwrite(f".\\imgs\\{saveName}{index}.png", hor*255)
            index += 1
        elif key == 27 or i > numTests:
            break
    cv2.destroyAllWindows()
    return

def testModelMultiplePrintEncoded(models, dataLoader, numTests = 30, shape=(3, 128, 128), save = False, saveName = ""):
    index = 1
    for i, (input, _) in enumerate(dataLoader):
        input = input.cuda(non_blocking= True)
        outputs = []
        all_encoded = []
        for model in models:
            encoded = model.encode(input)
            decoded = model.decode(encoded)
            outputs.append(decoded)
            all_encoded.append(encoded.cpu().detach().numpy().flatten())
        for e in zip(*all_encoded):
            print(e)

        input = input.view(shape).cpu().detach().numpy().transpose(1,2,0)
        outputs = [output.view(shape).cpu().detach().numpy().transpose(1,2,0) for output in outputs]
        outputs.insert(0, input)
        hor = np.concatenate(outputs, axis=1)
        hor = cv2.cvtColor(hor, cv2.COLOR_BGR2RGB)
        cv2.imshow("images", hor)

        key = cv2.waitKey(0)
        if  key == ord(' ') and save:
            cv2.imwrite(f".\\imgs\\{saveName}{index}.png", hor*255)
            index += 1
        elif key == 27 or i > numTests:
            break
    cv2.destroyAllWindows()
    return

def trainModel(model, dataloader, dataLoaderVal, criterion, scheduler,optimizer, epochs, save = True, savePath = ".\\models\\", name = "model", startEpoch = 0):
    print("[epoch : epoch p finished : running loss : loss difference : time]")
    model.cuda()
    model.train()
    runningLoss = 0
    lastLoss = 0
    for e in range(startEpoch, epochs):
        etime = time.time()
        rtime = time.time()
        for i, (input, _) in enumerate(dataloader):
            input = input.cuda(non_blocking = True)
            
            output = model(input)

            loss = criterion(output, input)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            runningLoss += loss.detach()

            if i % 100 == 0:
                runningLoss /=  100
                print("[%2d : %3.2f : %1.9f : %s%1.9f : %5.1f ]" % (
                        e,
                        ((i+1)*dataloader.batch_size / (len(dataloader) * dataloader.batch_size) * 100),
                        (runningLoss), ('+' if runningLoss > lastLoss else '-'),
                        (abs(runningLoss - lastLoss)),
                        (time.time()-rtime)
                    ))
                lastLoss = runningLoss
                runningLoss = 0
        if save:
            torch.save(model.state_dict(), savePath+name+f"-{e}.mld")
        print("Epoch time : %3.2f" % (time.time() - etime,))
        if scheduler is None:
            continue
        runningLoss = 0
        for i, (input, _) in enumerate(dataLoaderVal):
            input = input.cuda(non_blocking = True)
            
            output = model(input)

            loss = criterion(output, input)
            runningLoss += loss.item()

        runningLoss /= len(dataLoaderVal)
        scheduler.step(metrics=runningLoss)
        print("VAL : %2.5f" % (runningLoss))
        runningLoss = 0
    return model