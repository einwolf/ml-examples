#!/usr/bin/env python3

# Modified from
# https://clay-atlas.com/us/blog/2021/06/14/pytorch-en-build-gan-gnerate-mnist/
# https://github.com/ccs96307/PyTorch-GAN-Mnist

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import transforms

from model import discriminator, generator

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

start_time = time.time()
plt.rcParams['image.cmap'] = 'gray'


def show_images(filename, images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index + 1)
        plt.imshow(image.reshape(28, 28))


def save_images(filename, images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig, axs = plt.subplots(ncols=sqrtn, nrows=sqrtn, figsize=(6, 6), constrained_layout=True)

    for index, image in enumerate(images):
        row = index // sqrtn
        col = index % sqrtn

        axs[row, col].get_xaxis().set_visible(False)
        axs[row, col].get_yaxis().set_visible(False)
        axs[row, col].set_title(f'Image {index}')

        # axs[row, col].get_xaxis().set_ticks([])
        # axs[row, col].get_yaxis().set_ticks([])
        # axs[row, col].get_xaxis().set_label(f'Image {index}')

        with plt.ion():
            axs[row, col].imshow(image.reshape(28, 28))
            plt.pause(0.001)

    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def save_images2(filename, images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig, axs = plt.subplots(ncols=sqrtn, nrows=sqrtn, figsize=(6, 6), constrained_layout=True)

    for index, image in enumerate(images):
        row = index // sqrtn
        col = index % sqrtn

        axs[row, col].get_xaxis().set_visible(False)
        axs[row, col].get_yaxis().set_visible(False)
        axs[row, col].set_title(f'Image {index}')

        # axs[row, col].get_xaxis().set_ticks([])
        # axs[row, col].get_yaxis().set_ticks([])
        # axs[row, col].get_xaxis().set_label(f'Image {index}')

        with plt.ion():
            # Calling imshow is the only way to plot an image? No imdraw?
            axs[row, col].imshow(image.reshape(28, 28))
            # plt.pause(0.001)

    fig.savefig(filename, bbox_inches='tight')


# Discriminator Loss => BCELoss
def d_loss_function(inputs, targets):
    return nn.BCELoss()(inputs, targets)


def g_loss_function(inputs):
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    return nn.BCELoss()(inputs, targets)


# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Model
G = generator().to(device)
D = discriminator().to(device)
print(G)
print(D)

# Settings
epochs = 100
lr = 0.0002
batch_size = 64
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

# Load data
train_set = datasets.MNIST('../dataset/', train=True, download=True, transform=transform)
test_set = datasets.MNIST('../dataset/', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Train
for epoch in range(epochs):
    epoch += 1

    for times, data in enumerate(train_loader):
        times += 1
        real_inputs = data[0].to(device)
        test = 255 * (0.5 * real_inputs[0] + 0.5)

        real_inputs = real_inputs.view(-1, 784)
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        # Zero the parameter gradients
        d_optimizer.zero_grad()

        # Backward propagation
        d_loss = d_loss_function(outputs, targets)
        d_loss.backward()
        d_optimizer.step()

        # Generator
        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)

        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)

        g_loss = g_loss_function(fake_outputs)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if times % 100 == 0 or times == len(train_loader):
            # print('[{}/{}, {}/{}] D_loss: {:.3f} G_loss: {:.3f}'.format(epoch, epochs, times, len(train_loader), d_loss.item(),
            #                                                             g_loss.item()))
            print(f'[{epoch}/{epochs}, {times}/{len(train_loader)}] D_loss: {d_loss.item():.3f} G_loss: {g_loss.item():.3f}')

    imgs_numpy = (fake_inputs.data.cpu().numpy() + 1.0) / 2.0
    save_images(f'gan_epoch/images_epoch_{epoch:02d}', imgs_numpy[:16])
    #plt.show()

    if epoch % 50 == 0:
        torch.save(G, f'gan_epoch/generator_epoch_{epoch}.pth')
        print(f'Model saved epoch {epoch}.')

print('Training Finished.')
print('Cost Time: {}s'.format(time.time() - start_time))
