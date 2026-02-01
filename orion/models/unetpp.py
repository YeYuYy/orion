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

class UNetPlusPlus(on.Module):
    def __init__(self, in_channels=3, num_classes=10, features=[64, 128, 256, 512], upsample="bilinear"):
        super(UNetPlusPlus, self).__init__()
        
        self.features = features
        self.depth = len(features)
        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.nest_convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.cat = on.Cat()
        
        self.enc_convs.append(DoubleConv(in_channels, features[0]))
        for i in range(1, self.depth):
            self.pools.append(on.AvgPool2d(kernel_size=2, stride=2))
            self.enc_convs.append(DoubleConv(features[i-1], features[i]))

        for j in range(1, self.depth): # column
            col_convs = nn.ModuleList()
            col_ups = nn.ModuleList()
            
            for i in range(self.depth - j): # row                
                in_ch = features[i] * (j + 1)
                out_ch = features[i]
                col_convs.append(DoubleConv(in_ch, out_ch))
                if upsample == "bilinear":
                    up_layer = on.BilinearConv2d(features[i+1], features[i], kernel_size=3, stride=2, padding=1)
                elif upsample == "transconv":
                    up_layer = on.ConvTranspose2d(features[i+1], features[i], kernel_size=2, stride=2)
                else:
                    raise ValueError("Invalid upsample mode")
                col_ups.append(up_layer)
            
            self.nest_convs.append(col_convs)
            self.ups.append(col_ups)

        self.final_conv = on.Conv2d(features[0], num_classes, kernel_size=1)
        # self.argmax = on.Argmax()

    def forward(self, x):
        grid = [[] for _ in range(self.depth)]

        # ---------------------------
        # Step 1: Encoder (Backbone)
        # ---------------------------
        out = x
        for i in range(self.depth):
            if i > 0:
                out = self.pools[i-1](out)
            out = self.enc_convs[i](out)
            grid[i].append(out)

        # ---------------------------
        # Step 2: Nested Decoder
        # ---------------------------
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                skip_connections = grid[i][:j]
                up_feat = self.ups[j-1][i](grid[i+1][j-1])
                to_cat = skip_connections + [up_feat]
                concat_feat = self.cat(to_cat, dim=1)
                if self.he_mode:
                    if len(concat_feat) == 32:
                        concat_feat.ids = concat_feat.ids[:24]
                    elif len(concat_feat) == 16:
                        concat_feat.ids = concat_feat.ids[:12]
                out_conv = self.nest_convs[j-1][i](concat_feat)
                grid[i].append(out_conv)

        final_feature = grid[0][-1]
        logits = self.final_conv(final_feature)
        # pred = self.argmax(logits)
        
        return logits


def UNetPlusPlus_small(upsample="bilinear"):
    return UNetPlusPlus(features=[4, 8, 16], upsample=upsample)

def UNetPlusPlus_base(upsample="bilinear"):
    return UNetPlusPlus(features=[64, 128, 256, 512], upsample=upsample)

def UNetPlusPlus_large(upsample="bilinear"):
    return UNetPlusPlus(features=[96, 192, 384, 768], upsample=upsample)
