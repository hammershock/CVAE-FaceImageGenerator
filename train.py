# -*- coding: utf-8 -*-
# train.py
"""

NOTICE: PyTorch & torchvision REQUIRED!!!
pip install torch torchvision  # for model training
pip install tqdm (optional)  # for nice progress bars
(pip install pillow)

You Can find the CelebA dataset here:
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

1. A folder of aligned images
2. a txt annotation file of image labels

"""
from datetime import datetime, timedelta
import os
import sys

import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from main import CVAE, torch, nn


class CelebADataset(Dataset):
    def __init__(self, txt_file_path, img_dir, transform=None):
        self.data = []
        with open(txt_file_path, 'r') as f:
            for i, line in enumerate(f):
                if i <= 1:  # skip the counts and the header line
                    continue
                parts = line.split()
                filename = parts[0]
                label = np.array([int(p) for p in parts[1:]])
                self.data.append((filename, label))

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        img_name = os.path.join(self.img_dir, filename)
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        attributes = torch.from_numpy(label.astype('float')).float()

        return image, attributes


def vae_loss(recon_x, x, mu, log_var):
    batch_size = recon_x.size(0)
    # MSE Loss as Reconstruction Loss
    MSE = nn.functional.mse_loss(recon_x.view(batch_size, -1), x.view(batch_size, -1), reduction='sum')
    # KL divergence Loss, to measure the difference between:
    # the Latent Distribution the model actually learnt AND the Standard Normal Distribution
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = MSE + KLD
    return loss, np.array([loss.item(), MSE.item(), KLD.item()])


class Tqdm:
    def __init__(self, iterable, desc="Progress", bar_length=40):
        self.iterable = iterable
        self.desc = desc
        self.bar_length = bar_length
        self.start_time = datetime.now()
        self.total = len(self.iterable) if hasattr(self.iterable, "__len__") else None
        self.postfix = ""

    def __iter__(self):
        return self

    def __next__(self):
        if not hasattr(self, 'index'):
            self.index = 0
        if self.index < len(self.iterable):
            item = self.iterable[self.index]
            self.index += 1
            self.print_progress(self.index, item)
            return item
        else:
            sys.stdout.write('\n')
            sys.stdout.flush()
            raise StopIteration

    def print_progress(self, current_index, item):
        elapsed_time = datetime.now() - self.start_time
        percent = '?' if self.total is None else ((current_index / self.total) * 100)
        filled_length = 0 if self.total is None else int(round(self.bar_length * (current_index - 1) / self.total))
        bar = '#' * filled_length + '.' * (self.bar_length - filled_length)
        eta = elapsed_time / (current_index - 1) * (self.total - current_index + 1) if current_index > 1 and self.total is not None else timedelta(seconds=0)
        elapsed_str = str(elapsed_time).split('.')[0]  # Remove milliseconds part
        eta_str = str(eta).split('.')[0]

        progress_msg = f"\r{self.desc}: [{bar}] {percent:.1f}% Elapsed: {elapsed_str} ETA: {eta_str} {self.postfix}"
        sys.stdout.write(progress_msg)
        sys.stdout.flush()

    def set_postfix(self, **kwargs):
        postfix = ' '.join([f"{key}={value}" for key, value in kwargs.items()])
        self.postfix = postfix


if __name__ == '__main__':
    batch_size = 128
    epochs = 50
    lr = 0.001

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    MODEL_PATH = f'models/cvae_celeba.pth'

    # model & optimizer
    vae = CVAE(potential_dim=64, channels=3)
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # load the existing model
    try: vae.load_state_dict(torch.load(MODEL_PATH))
    except:pass

    # preprocess transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # dataset & dataloader
    train_dataset = CelebADataset(txt_file_path='data/celeba/list_attr_celeba.csv',
                                  img_dir='data/celeba/img_align_celeba',
                                  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # begin training
    p_bar = Tqdm(range(epochs))
    for epoch in p_bar:
        running_loss = np.array([0., 0., 0.])
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = vae(data, labels)
            loss, losses = vae_loss(recon_batch, data, mu, log_var)
            loss.backward()

            optimizer.step()
            running_loss += 1 / (batch_idx + 1) * (losses - running_loss)
            p_bar.set_postfix(progress=f'{(batch_idx + 1) / len(train_loader) * 100:.2f}%',
                              totalLoss=f'{running_loss[0]:.3f}', MSELoss=f'{running_loss[1]:.3f}',
                              KLDLoss=f'{running_loss[2]:.3f}')

        torch.save(vae.state_dict(), MODEL_PATH)
