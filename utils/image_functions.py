from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def show_images(images, n_max=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images.detach()[:n_max], nrow=8).permute(1, 2, 0))


def show_batch(dl, n_max=64):
    for images, _ in dl:
        show_images(images, n_max)
        break
