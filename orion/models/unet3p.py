import torch
import torch.nn as nn
import orion.nn as on

class DoubleConv(on.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            on.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            on.BatchNorm2d(out_channels),
            on.ReLU(),
            on.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            on.BatchNorm2d(out_channels),
            on.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UNet3Plus(on.Module):
    def __init__(self, in_channels=3, num_classes=10, features=[64, 128, 256, 512, 1024], upsample="bilinear"):
        super(UNet3Plus, self).__init__()
        
        self.features = features
        self.depth = len(features)
        self.cat_channels = features[0]
        self.upsample_mode = upsample

        # ----------------------------------------------------
        # 1. Encoder (Backbone)
        # ----------------------------------------------------
        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        self.enc_convs.append(DoubleConv(in_channels, features[0]))
        for i in range(1, self.depth):
            self.pools.append(on.AvgPool2d(kernel_size=2, stride=2))
            self.enc_convs.append(DoubleConv(features[i-1], features[i]))

        # ----------------------------------------------------
        # 2. Decoder with Full-Scale Skip Connections
        # ----------------------------------------------------
        
        self.dec_connections = nn.ModuleList()
        self.dec_fusions = nn.ModuleList()
        for i in range(self.depth - 2, -1, -1):
            connections = nn.ModuleList()
            for k in range(self.depth):
                if k > i:
                    scale = 2 ** (k - i)
                    in_ch = features[0] * self.depth if k < self.depth -1 else features[-1]
                    if self.upsample_mode == "bilinear":
                        up = on.BilinearConv2d(in_ch, self.cat_channels, kernel_size=3, stride=scale, padding=1)
                    elif self.upsample_mode == "transconv":
                         up = on.ConvTranspose2d(in_ch, self.cat_channels, kernel_size=scale, stride=scale)
                    connections.append(up)
                elif k == i:
                    conv = on.Conv2d(features[k], self.cat_channels, kernel_size=3, padding=1, bias=False)
                    connections.append(nn.Sequential(conv, on.BatchNorm2d(self.cat_channels), on.ReLU()))
                elif k < i:
                    scale = 2 ** (i - k)
                    pool = on.AvgPool2d(kernel_size=scale, stride=scale)
                    conv = on.Conv2d(features[k], self.cat_channels, kernel_size=3, padding=1, bias=False)
                    connections.append(nn.Sequential(
                        pool, 
                        conv, 
                        on.BatchNorm2d(self.cat_channels), 
                        on.ReLU()
                    ))

            self.dec_connections.append(connections)
            total_in_ch = self.cat_channels * self.depth
            self.dec_fusions.append(DoubleConv(total_in_ch, total_in_ch))

        self.cat = on.Cat()        
        self.final_conv = on.Conv2d(self.cat_channels * self.depth, num_classes, kernel_size=1)

    def forward(self, x):
        enc_results = []
        out = x
        for i in range(self.depth):
            if i > 0:
                out = self.pools[i-1](out)
            out = self.enc_convs[i](out)
            enc_results.append(out)

        dec_results = {}
        dec_results[self.depth - 1] = enc_results[-1]
        for idx, i in enumerate(range(self.depth - 2, -1, -1)):
            current_layer_inputs = []
            connections = self.dec_connections[idx]
            for k in range(self.depth):
                if k < i:
                    src = enc_results[k]
                elif k == i:
                    src = enc_results[k]
                else:
                    src = dec_results[k]
                processed = connections[k](src)
                current_layer_inputs.append(processed)
            cat_out = self.cat(current_layer_inputs, dim=1)
            fusion_out = self.dec_fusions[idx](cat_out)
            dec_results[i] = fusion_out

        final_feature = dec_results[0]
        logits = self.final_conv(final_feature)
        
        return logits
    

def UNet3Plus_small(upsample="bilinear"):
    return UNet3Plus(features=[4, 8, 16, 32], upsample=upsample)

def UNet3Plus_base(upsample="bilinear"):
    return UNet3Plus(features=[64, 128, 256, 512, 1024], upsample=upsample)
