from CustomImageDataset import CustomImageDataset
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()
        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model_out = nn.Sequential(*[nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False), nn.Softmax()])
        self.upsample = nn.Upsample(size=256)

    def forward(self, x):
        conv1_2 = self.model1(x)
        conv2_2 = self.model2(conv1_2)
        out_reg = self.model_out(conv2_2)
        return self.upsample(out_reg)
    


