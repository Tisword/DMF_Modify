from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def read_img_L(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process.
    用于读取深度图片
    """
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('L')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid,img_path.split('/')[-1]

#这里读两个数据集
class ImageDataset_cross(Dataset):
    def __init__(self, dataset1, dataset2, transform=None):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length_dataset2 = len(self.dataset2)
        self.transform = transform

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        img1_path, pid, camid, trackid = self.dataset1[index]
        img2_path, _, _, _ = self.dataset2[random.randint(0,self.length_dataset2-1)]
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, pid, camid, trackid, img1_path.split('/')[-1], img2_path.split('/')[-1]


class ImageDataSet_Mutil(Dataset):
    def __init__(self, dataset1, dataset2, transformRGB=None,transformDepth=None):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.transformRGB = transformRGB
        self.transformDepth=transformDepth

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        img1_path, pid, camid,trackid= self.dataset1[index]
        img2_path, _, _,_= self.dataset2[index]
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        if self.transformRGB is not None:
            img1 = self.transformRGB(img1)
        if self.transformDepth is not None:
            img2 = self.transformDepth(img2)
        return img1, img2, pid, camid, trackid,img1_path.split('/')[-1], img2_path.split('/')[-1]