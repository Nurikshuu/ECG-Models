"""
InceptionTime baseline architecture extracted from
notebooks/notebook_baseline/train_inception_pro (1).ipynb.
"""

import torch
import torch.nn.functional as F


class InceptionModule(torch.nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[10, 20, 40],
                 bottleneck_channels=32, use_bottleneck=True):
        super().__init__()
        if use_bottleneck and in_channels > 1:
            self.bottleneck = torch.nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
            conv_in = bottleneck_channels
        else:
            self.bottleneck = None
            conv_in = in_channels

        self.convs = torch.nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(
                torch.nn.Conv1d(conv_in, n_filters, ks, padding='same', bias=False))

        self.maxpool = torch.nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = torch.nn.Conv1d(in_channels, n_filters, 1, bias=False)
        self.bn = torch.nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))

    @property
    def out_channels(self):
        return self.bn.num_features

    def forward(self, x):
        x_bn = self.bottleneck(x) if self.bottleneck is not None else x
        conv_outs = [conv(x_bn) for conv in self.convs]
        pool_out = self.conv_pool(self.maxpool(x))
        out = torch.cat(conv_outs + [pool_out], dim=1)
        return F.relu(self.bn(out), inplace=True)


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[10, 20, 40],
                 bottleneck_channels=32, depth=3):
        super().__init__()
        modules = []
        ch_in = in_channels
        for _ in range(depth):
            m = InceptionModule(ch_in, n_filters, kernel_sizes, bottleneck_channels)
            modules.append(m)
            ch_in = m.out_channels
        self.inception_modules = torch.nn.Sequential(*modules)
        self.out_ch = ch_in
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, self.out_ch, 1, bias=False),
            torch.nn.BatchNorm1d(self.out_ch))

    def forward(self, x):
        out = self.inception_modules(x)
        return F.relu(out + self.shortcut(x), inplace=True)


class InceptionTimeEncoder(torch.nn.Module):
    def __init__(self, n_filters=32, kernel_sizes=[10, 20, 40],
                 bottleneck_channels=32, n_blocks=2, depth_per_block=3):
        super().__init__()
        blocks = []
        in_ch = 1
        for _ in range(n_blocks):
            block = InceptionBlock(in_ch, n_filters, kernel_sizes,
                                   bottleneck_channels, depth_per_block)
            blocks.append(block)
            in_ch = block.out_ch
        self.blocks = torch.nn.Sequential(*blocks)
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.d_model = in_ch

    def forward(self, x):
        B, n_leads, L = x.shape
        x = x.reshape(B * n_leads, 1, L)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return x.reshape(B, n_leads, self.d_model)


class InceptionTimeBaseline(torch.nn.Module):
    def __init__(self, num_classes=5, n_filters=32, kernel_sizes=[10, 20, 40],
                 bottleneck_channels=32, n_blocks=2, depth_per_block=3,
                 head_dropout=0.5):
        super().__init__()
        self.encoder = InceptionTimeEncoder(n_filters, kernel_sizes,
                                            bottleneck_channels, n_blocks,
                                            depth_per_block)
        d = self.encoder.d_model
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(head_dropout),
            torch.nn.Linear(d, num_classes))

    def forward(self, x):
        features = self.encoder(x)
        features = features.mean(dim=1)
        return self.classifier(features)

