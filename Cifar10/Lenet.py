import torch
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(32*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        print('conv out:', out.shape)

    def forward(self, x): # 前向传播过程，网络非常浅，只包含两个卷积层和三个全连接层
        batchsz = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batchsz, 32*5*5)
        logits = self.fc_unit(x)
        return logits

def main():
    net = Lenet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('lenet out:', out.shape)

if __name__ == '__main__':
    main()