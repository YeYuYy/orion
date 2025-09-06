import torch.nn as nn
import orion.nn as on


class InvBottleneck(on.Module):
    def __init__(self, Ci, Co, stride=1, expansion=6):
        super().__init__()
        self.conv1 = on.Conv2d(Ci, Ci * expansion, kernel_size=1, bias=False)
        self.bn1   = on.BatchNorm2d(Ci * expansion)
        self.act1  = on.ReLU6(degree=127)

        self.conv2 = on.Conv2d(Ci * expansion, Ci * expansion, kernel_size=3, 
                               stride=stride, groups=Ci * expansion, padding=1, bias=False)
        self.bn2   = on.BatchNorm2d(Ci * expansion)
        self.act2  = on.ReLU6(degree=127)

        self.conv3 = on.Conv2d(Ci * expansion, Co, kernel_size=1, stride=1, bias=False)
        self.bn3   = on.BatchNorm2d(Co)

        self.add = on.Add()
        self.add_residual = (stride == 1 and Ci == Co)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.add_residual:
            out = self.add(out, x)
        return out


class MobileNetV2(on.Module):
    def __init__(self, dataset='cifar10'):
        super().__init__()

        first_stride, num_classes = get_resnet_config(dataset)
        num_blocks = [1, 2, 3, 4, 3, 3, 1]
        num_chans = [32, 16, 24, 32, 64, 96, 160, 320]
        strides = [1, 2, 2, 2, 1, 2, 1]

        self.conv1 = on.Conv2d(3, 32, 3, stride=first_stride, padding=1, bias=False)
        self.bn1   = on.BatchNorm2d(32)
        self.act1  = on.ReLU6(degree=127)
        
        self.layers = nn.ModuleList()
        for i in range(len(num_blocks)):
            blocks = []
            for j in range(num_blocks[i]):
                stride = strides[i] if j == 0 else 1
                input_chans = num_chans[i] if j == 0 else num_chans[i+1]
                blocks.append(InvBottleneck(input_chans, num_chans[i+1], stride=stride))
            self.layers.append(nn.Sequential(*blocks))
            
        self.conv2 = on.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2   = on.BatchNorm2d(1280)
        self.act2  = on.ReLU6(degree=127)
        self.avgpool = on.AdaptiveAvgPool2d(output_size=(1,1)) 
        self.flatten = on.Flatten()
        self.linear  = on.Linear(1280, num_classes)
    
    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)

        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.linear(out)
    

def get_resnet_config(dataset):
    configs = {
        "cifar10": {"first_stride": 1, "num_classes": 10},
        "cifar100": {"first_stride": 1, "num_classes": 100},
        "tiny": {"first_stride": 2, "num_classes": 200},
        "imagenet": {"first_stride": 2, "num_classes": 1000},
    }

    if dataset not in configs:
        raise ValueError(f"ResNet with dataset {dataset} is not supported.")
    config = configs[dataset]
    
    return config["first_stride"], config["num_classes"]


if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    net = MobileNetV2()
    net.eval()

    x = torch.randn(1,3,224,224)
    total_flops = FlopCountAnalysis(net, x).total()

    summary(net, (3,224,224), device="cpu")
    print("Total flops: ", total_flops)
    