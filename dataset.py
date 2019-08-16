"""
Classes for the Datasets
"""

import os
from xml.etree import ElementTree
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class VOCCustom(Dataset):
    """
    Creates the VOC 2012 Dataset with certain classes
    and target as another transformed image.
    """
    YEAR = 'VOC2012'
    DEFAULT_PATH = ['VOCdevkit', 'VOC2012', 'ImageSets', 'Main']
    CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
               'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike',
               'train', 'bottle', 'chair', 'diningtable', 'pottedplant',
               'sofa', 'tvmonitor']

    def __init__(self, root_dir="./data/pascal_voc/", img_transform=None,
                 target_transform=None, classes='all', image_set='train'):
        """
        Args:
            root_dir: Root directory of the images.
            img_transform: Tranformation applied to the images.
            target_transform: Tranformation applied to the target image.
            classes: Classes used in this dataset, a list of classes names.
                     'all': all classes are used.
            image_set: 'train', 'val' or 'trainval'.

        Return:
            (image, target): Tuple with the image and target.
        """
        # Attributes
        self.root_dir = root_dir
        self.image_set = image_set
        self.i_transform = img_transform
        self.t_transform = target_transform
        if classes == 'all':
            self.classes = self.CLASSES
        else:
            self.classes = classes
        self.classes_id = {cls: i for i, cls in enumerate(self.classes)}

        # Load images
        self.images = self._get_images_list()

    def _get_images_list(self,):
        """
        List of images present in the classes used.
        """
        main_dir = os.path.join(self.root_dir, *self.DEFAULT_PATH)
        # For each class
        images = []
        for c in self.classes:
            file_path = os.path.join(main_dir,
                                     c + '_' + self.image_set + '.txt')
            with open(file_path) as f:
                files = f.readlines()
                imgs = [line.split(' ')[0]
                        for line in files if line[-3] != '-']
            images += imgs
        return list(set(images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if type(idx) == torch.Tensor:
            idx = idx.item()
        # Load images
        img_path = os.path.join(self.root_dir,
                                'VOCdevkit',
                                'VOC2012',
                                'JPEGImages',
                                self.images[idx] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        img_t = Image.open(img_path).convert('RGB')
        # Transforms
        img = self.i_transform(img)
        img_t = self.t_transform(img_t)

        # Output
        return (img, img_t)


def show_image(img, ax=None):
    """
    Show Image in the path variable.
    """
    if isinstance(img, str):
        image = Image.open(img)
    elif isinstance(img, torch.Tensor):
        image = T.ToPILImage()(img)
    else:
        image = img
    if ax is None:
        f = plt.figure(figsize=(12, 10))
        ax = f.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.imshow(image)
    return ax

# Prepared Watermark
wm = Image.open('wm_ready.png')


def add_watermark(img, wm=wm):
    """
    Adds watermark to an image
    """
    c_img = img.copy()
    c_img.paste(wm, mask=wm)
    return c_img


if __name__ == "__main__":
    cls_test = ['bicycle', 'bus', 'car', 'motorbike']
    SIZE = (300, 400)
    TRANSFORM_I = T.Compose([T.Resize(SIZE),
                             T.ToTensor()])
    TRANSFORM_T = T.Compose([T.Resize(SIZE),
                             add_watermark,
                             T.ToTensor()])
    ds = VOCCustom('D:\DATASETS\VOCDetection', classes=cls_test,
                   img_transform=TRANSFORM_I, target_transform=TRANSFORM_T)
    iter_ds = iter(ds)
    for i in range(5):
        img, img_t = next(iter_ds)
        show_image(img)
        plt.show()
        show_image(img_t)
        plt.show()
