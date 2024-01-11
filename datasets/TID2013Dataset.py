import os
import re

from PIL import Image
import torchvision.transforms as tt
from torch.utils.data import Dataset


class TID2013Dataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform if transform is not None else tt.ToTensor()
        self.images = []
        self.reference_images = []

        self._load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.reference_images[idx // (24 * 5)]

    def _load_data(self):
        references_path = os.path.join(self.root, 'reference_images')
        images_path = os.path.join(self.root, 'distorted_images')

        if not os.path.exists(references_path) or not os.path.exists(images_path):
            raise ValueError('Dataset not found. Please download it from https://www.ponomarenko.info/tid2013.htm')

        def __sort_key(filename):
            match = re.match(r'i(\d+)_(\d+)_(\d+)\.bmp', filename.lower())
            return int(match.group(1)), int(match.group(2)), int(match.group(3))

        reference_images = sorted([file for file in os.listdir(references_path) if file.lower().endswith('.bmp')])
        distorted_images = sorted(
            [file for file in os.listdir(images_path) if file.lower().endswith('.bmp')],
            key=__sort_key
        )

        if len(reference_images) != 25:
            raise ValueError('Expected 25 reference images, found {}'.format(len(reference_images)))

        if len(distorted_images) != 3000:
            raise ValueError('Expected 3000 distorted images, found {}'.format(len(distorted_images)))

        for reference_image in reference_images:
            image_path = os.path.join(references_path, reference_image)
            with open(image_path, 'rb') as f:
                self.reference_images.append(self.transform(Image.open(f).convert('RGB')))

        for image in distorted_images:
            image_path = os.path.join(images_path, image)
            with open(image_path, 'rb') as f:
                self.images.append(self.transform(Image.open(f).convert('RGB')))
