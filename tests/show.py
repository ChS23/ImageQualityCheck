import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from datasets.TID2013Dataset import TID2013Dataset

root = r'C:\Users\c4s23\YandexDisk\GitHub\ImageQualityCheck\dataset\TID2013'
dataset = TID2013Dataset(root, transform=None)


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images.detach()[:nmax], nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


dataloader = DataLoader(dataset, batch_size=24 * 5, shuffle=False)

distorted_image, reference_image = next(iter(dataloader))

show_images(distorted_image, 16)

plt.show()
