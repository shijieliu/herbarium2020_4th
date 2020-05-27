from torch.utils import data
from torch.utils.data import DataLoader
from legacy.focalloss import *
from torchvision import transforms as T
import jsonlines
import nori2 as nori
import cv2
import pickle
import random
from PIL import Image
from typing import Iterator, List, Optional, Union
from prefetch_generator import BackgroundGenerator


class BinaryCrop(object):
    def __init__(self):
        pass

    def __call__(self, img):
        randint = random.randint(0, 2)
        w, h = img.size
        if randint == 0:
            return img
        if randint == 1:
            return img.crop((0, 0, w, h // 2))
        return img.crop((0, h // 2, w, h))


def get_transforms(cfg):
    transforms = []
    for transform_pack in cfg:
        transform_pack = eval(transform_pack)
        transform_typ, transform_args = transform_pack[0], transform_pack[1]
        if transform_typ == 'binary_crop':
            transforms.append(BinaryCrop())
        elif transform_typ == 'random_rotation':
            print(transform_args)
            transforms.append(T.RandomRotation(transform_args))
        elif transform_typ == 'resize':
            transforms.append(T.Resize(transform_args))
        elif transform_typ == 'to_tensor':
            transforms.append(T.ToTensor())
        elif transform_typ == 'normalize':
            transforms.append(T.Normalize(*transform_args))
        else:
            raise NotImplementedError('transform type %s not implemented' % transform_typ)
    return transforms



def make_weights_for_balanced_classes(images, nclasses=32093):
    count = [0] * nclasses
    for item in images:
        category_id = int(item['category_id'])
        if category_id > 23079:
            category_id -= 1
        count[category_id] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, item in enumerate(images):
        category_id = int(item['category_id'])
        if category_id > 23079:
            category_id -= 1
        weight[idx] = weight_per_class[category_id]
    return weight


class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, weights, shuffle=True):
        super(DistributedWeightedSampler, self).__init__(dataset, shuffle=shuffle)
        self.weights = weights

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # subweights
        weights = self.weights[indices]
        assert len(indices) == self.num_samples
        return iter(torch.multinomial(weights, self.num_samples, True).tolist())


class HerbariumDS(data.Dataset):
    def __init__(self, filename, transform_cfg):
        self.nr = nori.Fetcher()
        self.data = []
        with jsonlines.open(filename, 'r') as f:
            for line in f:
                self.data.append(line)
        self.transforms = T.Compose(get_transforms(transform_cfg))

    def __getitem__(self, index):
        meta = self.data[index]
        img_path = meta['image_path']
        label = meta['category_id']
        if label > 23079:
            label -= 1
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.data)


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
