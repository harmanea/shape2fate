import numpy as np
import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class DownSamplingLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ConvLayer(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)


class UpSamplingLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = 'transposed'):
        """
        :param mode: 'transposed' for transposed convolution, or 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        """
        super().__init__()

        if mode == 'transposed':
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        elif mode in {'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'}:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
        else:
            raise ValueError(f'Unsupported mode [{mode}], supported modes are "transposed", "nearest", "linear", "bilinear", "bicubic" or "trilinear"')

        self.conv = ConvLayer(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, depth: int = 3, start_filters: int = 16, up_mode: str = 'transposed'):
        super().__init__()

        self.inc = ConvLayer(in_channels, start_filters)

        # Contracting path
        self.down = nn.ModuleList([DownSamplingLayer(start_filters * 2 ** i, start_filters * 2 ** (i + 1))
                                   for i in range(depth)])

        # Expansive path
        self.up = nn.ModuleList([UpSamplingLayer(start_filters * 2 ** (i + 1), start_filters * 2 ** i, up_mode)
                                 for i in range(depth - 1, -1, -1)])

        self.outc = nn.Conv2d(start_filters, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)

        outputs = []

        for module in self.down:
            outputs.append(x)
            x = module(x)

        for module, output in zip(self.up, reversed(outputs)):
            x = module(x, output)

        return self.outc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, cnn_filters, pe_dropout, pe_max_len, n_attention_heads, dim_feedforward, num_transformer_layers, fc_dropout):
        super(TransformerModel, self).__init__()

        self.cnn_filters = cnn_filters

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_filters[0], cnn_filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_filters[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_filters[1], cnn_filters[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_filters[1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )

        self.pe = PositionalEncoding(cnn_filters[1], pe_dropout, pe_max_len)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=cnn_filters[1],
                                                     nhead=n_attention_heads,
                                                     dim_feedforward=dim_feedforward,
                                                     batch_first=True,
                                                     norm_first=True),
            num_layers=num_transformer_layers,
            norm=nn.LayerNorm(cnn_filters[1]),
        )

        self.dropout = nn.Dropout(fc_dropout)

        self.fc = nn.Linear(cnn_filters[1], 1)

    def forward(self, x):
        b, s, h, w = x.shape

        x = x.reshape(b * s, 1, h, w)  # from (batch, seq, height, width) to (batch * seq, channels, height, width)
        x = self.cnn(x)  # return shape (batch * seq, channels, 1, 1)
        x = x.reshape(b, s, self.cnn_filters[1])  # return shape (batch, seq, channels)
        x = self.pe(x)
        x = self.transformer(x)  # return shape (batch, seq, channels)
        x = x.mean(1)  # pool sequence to get (batch, channels)
        x = self.dropout(x)
        x = self.fc(x)  # return shape (batch, 1)
        x = x[..., 0]  # squeeze the last dimension

        return x
