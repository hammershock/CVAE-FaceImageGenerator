import os

import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image

import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from main import CVAE, torch, nn


class CelebADataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.attr_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.attr_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.attr_frame.iloc[idx, 0])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        attributes = self.attr_frame.iloc[idx, 1:].to_numpy()
        attributes = torch.from_numpy(attributes.astype('float')).float()

        return image, attributes


def vae_loss(recon_x, x, mu, log_var):
    # 重构损失：通常使用二元交叉熵（BCE）损失
    MSE = nn.functional.mse_loss(recon_x.view(-1, 32 * 32), x.view(-1, 32 * 32), reduction='sum')
    # KL 散度损失：用于度量学到的潜在分布与标准正态分布之间的差异
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = MSE + KLD
    return loss, np.array([loss.item(), MSE.item(), KLD.item()])


if __name__ == '__main__':
    batch_size = 128
    epochs = 50
    lr = 0.001

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    MODEL_PATH = f'models/cvae_celeba.pth'

    # 模型和优化器
    vae = CVAE(potential_dim=64, channels=3)
    vae.to(device)

    try: vae.load_state_dict(torch.load(MODEL_PATH))
    except:pass

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = CelebADataset(csv_file='data/celeba/list_attr_celeba.csv',
                                  img_dir='data/celeba/img_align_celeba/img_align_celeba',
                                  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # writer = SummaryWriter('runs/cvae_celeba')

    p_bar = tqdm(range(epochs))
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
