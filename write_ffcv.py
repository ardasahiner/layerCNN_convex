from typing import List

import torch as ch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

datasets = {
    'train': torchvision.datasets.CIFAR10('/mnt/dense/sahiner', train=True, download=True),
    'test': torchvision.datasets.CIFAR10('/mnt/dense/sahiner', train=False, download=True)
}

for (name, ds) in datasets.items():
    writer = DatasetWriter(f'/mnt/dense/sahiner/cifar_{name}.beton', {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(ds)
