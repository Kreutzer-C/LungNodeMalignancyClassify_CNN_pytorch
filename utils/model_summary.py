import torch
from torchsummary import summary
from model.model import ConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=ConvNet(2,0.15).to(device)
summary(model,input_size=(1,64,64))