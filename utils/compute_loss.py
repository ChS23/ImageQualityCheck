from tqdm import tqdm


def compute_loss(dataloader, loss_fn, device):
    total_loss = 0.0
    num_batches = len(dataloader)

    for images, ref_images in tqdm(dataloader):
        images = images.to(device)
        ref_images = ref_images.to(device)

        loss = loss_fn(images, ref_images)
        total_loss += loss.item()

    return total_loss / num_batches
