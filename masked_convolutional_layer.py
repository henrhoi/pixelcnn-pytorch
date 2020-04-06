import torch.nn as nn
import torch


class MaskedConv2d(nn.Conv2d):
    """
    Class extending nn.Conv2d to use masks.
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding=0):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding)
        self.register_buffer('mask', torch.ones(out_channels, in_channels, kernel_size, kernel_size).float())

        # _, depth, height, width = self.weight.size()
        h, w = kernel_size, kernel_size

        if mask_type == 'A':
            self.mask[:, :, h // 2, w // 2:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0
        else:
            self.mask[:, :, h // 2, w // 2 + 1:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
