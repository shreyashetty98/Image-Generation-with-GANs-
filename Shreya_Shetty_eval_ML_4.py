import torch
import h5py
import matplotlib.pyplot as plt
import numpy as np
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z.view(-1, z_dim, 1, 1))
        return out[:, :, 2:30, 2:30]


def load_model(model, filepath):
    with h5py.File(filepath, 'r') as h5file:
        for name, param in model.named_parameters():
            param.data = torch.tensor(h5file[name][:])


def plot_images(gen, num_gen=25, z_dim=100):
    z = torch.randn(num_gen, z_dim)
    gen_images = gen(z).detach().numpy()

    # Plotting the images
    h, w = 28, 28
    n = int(np.sqrt(num_gen))
    I_gen = np.empty((h * n, w * n))

    for i in range(n):
        for j in range(n):
            I_gen[i * h:(i + 1) * h, j * w:(j + 1) * w] = gen_images[i * n + j, 0]

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(I_gen, cmap='gray')
    plt.show()


if __name__ == "__main__":
    z_dim = 100
    h5_filepath = "generator.h5"
    num_gen = 25
    model = Generator(z_dim)
    load_model(model, h5_filepath)
    plot_images(model, num_gen=num_gen, z_dim=z_dim)
