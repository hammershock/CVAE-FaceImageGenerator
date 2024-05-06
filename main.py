# -*- coding: utf-8 -*-
"""
A NumPy implementation of the Conditional Convolutional Variational Autoencoder (CVAE), trained on the Celeba Faces dataset.
A lightweight framework, utilizes only NumPy for numerical forward inference

requirements:
ONLY NumPy and matplotlib

@author: hammershock
@email: hammershock@163.com
@date: 2024.5.3
"""
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    import NumPyTorch.nn as nn
    import NumPyTorch as torch
else:
    import torch.nn as nn
    import torch


class CVAE(nn.Module):
    """
    Conditional Convolutional VAE (Variational Autoencoder)
    """

    def __init__(self, potential_dim, channels, num_attributes=40):
        super(CVAE, self).__init__()
        self.potential_dim = potential_dim
        self.channels = channels

        # Linear layer for encoding the class labels
        self.attr_embedding = nn.Linear(num_attributes, num_attributes)

        output_shape = (128, 6, 7)

        output_dim = output_shape[0] * output_shape[1] * output_shape[2]
        # image_size = (178, 218)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels + num_attributes, 64, kernel_size=3, stride=2, padding=1),  # 89
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 45
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 23
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 12
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 6
            nn.Flatten(),
        )

        self.enc_mu = nn.Linear(output_dim, potential_dim)  # Mean
        self.enc_log_var = nn.Linear(output_dim, potential_dim)  # Log Variance
        # Decoder
        self.decoder_fc = nn.Linear(potential_dim + num_attributes, output_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, output_shape),  # 6
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # 12
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),  # 23
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),  # 45
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),  # 89
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.AdaptiveAvgPool2d((178, 218)),
            nn.Sigmoid()
        )

    def encode(self, x, attributes):
        # Embed the labels to the same dimension as the image
        embedded_attrs = self.attr_embedding(attributes)
        embedded_attrs = torch.unsqueeze(embedded_attrs, 2)
        embedded_attrs = torch.unsqueeze(embedded_attrs, 3)
        size = (embedded_attrs.size(0), embedded_attrs.size(1), x.size(2), x.size(3))
        embedded_attrs = torch.expand_copy(embedded_attrs, size)

        # Concatenate the labels and image
        x = torch.cat((x, embedded_attrs), dim=1)

        # Pass through the encoder
        x = self.encoder(x)
        mu = self.enc_mu(x)
        log_var = self.enc_log_var(x)

        # Reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, log_var

    def decode(self, z, labels):
        # Embed the labels and concatenate with the latent vector
        labels = self.attr_embedding(labels)
        z = torch.cat((z, labels), dim=1)

        # Pass through the decoder
        x = self.decoder_fc(z)
        x = self.decoder(x)
        return x

    def forward(self, x, labels):
        z, mu, log_var = self.encode(x, labels)
        reconstructed_x = self.decode(z, labels)
        return reconstructed_x, mu, log_var


if __name__ == "__main__":
    model = CVAE(64, 3)

    state_dict = torch.load('models/cvae_celeba_decoder_only_fp16.pkl')  # fake torch, just load the .pkl file
    # If you don't mind installing the real PyTorch, you can convert the pth model to a fake PyTorch model using it
    # torch.load_pytorch('models/cvae_celeba.pth')
    # torch.save(state_dict, f'models/cvae_celeba.pkl')
    # Save state_dict in fake PyTorch format
    model.load_state_dict(state_dict)
    # exit(0)
    # np.random.seed(27)

    # 40 labels
    labels = [-1, -1, -1, 1, 1,    # "五点钟胡须", "挑眉", "魅力", "眼袋", "秃头"
              -1, 1, -1, -1, -1,   # "刘海", "厚嘴唇", "大鼻子", "黑色头发", "金色头发"
              1, 1, -1, 1, -1,     # "模糊的", "棕色头发", "浓眉", "圆润", "双下巴"
              -1, -1, -1, 1, -1,  # "眼镜", "山羊胡", "灰白头发", "浓妆", "高颧骨"
              -1, 1, -1, -1, 1,     # "男性", "微张嘴", "小胡子", "狭长眼睛", "无胡子"
              -1, 1, -1, -1, -1,   # "椭圆脸型", "苍白肤色", "尖鼻子", "发际线后退", "红润脸颊"
              1, 1, 1, -1, 1,    # "鬓角", "微笑", "直发", "波浪发型", "耳环"
              -1, -1, 1, 1, 1]     # "戴帽子", "口红", "项链", "领带", "年轻"
    # or use random labels
    labels = np.sign(np.random.randn(1, 40))
    labels = np.array(labels).reshape(1, -1)
    hidden_z = np.random.randn(1, 64)

    labels = torch.FloatTensor(labels)
    hidden_z = torch.FloatTensor(hidden_z)
    result = model.decode(hidden_z, labels)

    # Display the image, use matplotlib
    plt.imshow(result.squeeze(0).transpose(1, 2, 0))
    plt.axis('off')  # Hide the axis
    plt.show()
