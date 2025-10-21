import torch
import torch.nn as nn
import orion.nn as on


class TestPermutation(on.Module):
    def __init__(self):
        super(TestPermutation, self).__init__()
        self.perm1 = on.Permutation([0, 2, 3, 1])
        self.perm2 = on.Permutation([0, 3, 1, 2], target_gap=4)
        self.perm3 = on.Permutation([0, 1, 2, 3], target_gap=2)
        self.perm4 = on.Permutation([0, 2, 3, 1], target_gap=1)
        self.perm5 = on.Permutation([0, 1, 2, 3])

    def forward(self, x):
        return self.perm5(self.perm4(self.perm3(self.perm2(self.perm1(x)))))
    

class TestCatPerm(on.Module):
    def __init__(self):
        super(TestCatPerm, self).__init__()
        self.conv0 = on.Conv2d(16, 4, kernel_size=3, padding=1, stride=2)
        self.conv1 = on.Conv2d(16, 4, kernel_size=3, padding=1, stride=2)
        self.perm0 = on.Permutation([0, 3, 2, 1], target_gap=1)
        self.cat = on.Cat()

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        y = self.cat([self.perm0(x0), self.perm0(x1)], dim=1)
        return y
    

class TestConvPerm(on.Module):
    def __init__(self):
        super(TestConvPerm, self).__init__()
        self.perm = on.Permutation([0, 1, 2, 3], target_gap=2)
        self.conv = on.ConvTranspose2d(16, 8, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.perm(x))
    