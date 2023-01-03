# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class TVPR2(BaseImageDataset):
    """

    """
    dataset_dir1 = 'color'
    dataset_dir2 = 'depth'
    common_dir = 'market1501'
    def __init__(self, root=r'/data3/QK/TransREID/TransReID-main/data', verbose=True, **kwargs):
        super(TVPR2, self).__init__()
        self.dataset_dir1 = osp.join( osp.join(root, self.dataset_dir1),self.common_dir)
        self.dataset_dir2=osp.join( osp.join(root, self.dataset_dir2),self.common_dir)
        self.train1_dir = osp.join(self.dataset_dir1, 'bounding_box_train')
        self.train2_dir = osp.join(self.dataset_dir2, 'bounding_box_train')
        self.query1_dir = osp.join(self.dataset_dir1, 'query')
        self.query2_dir = osp.join(self.dataset_dir2, 'query')
        self.gallery1_dir = osp.join(self.dataset_dir1, 'bounding_box_test')
        self.gallery2_dir = osp.join(self.dataset_dir2, 'bounding_box_test')

        self._check_before_run()

        #train1 和train2 这个时候是要对应起来的
        train1 = self._process_dir(self.train1_dir, relabel=True)
        train2 = self._process_dir(self.train2_dir, relabel=True)
        query1 = self._process_dir(self.query1_dir, relabel=False)
        query2= self._process_dir(self.query2_dir, relabel=False)
        gallery1 = self._process_dir(self.gallery1_dir, relabel=False)
        gallery2=self._process_dir(self.gallery2_dir, relabel=False)

        if verbose:
            print("=> TVPR2 loaded")
            self.print_dataset_statistics(train1, query1, gallery1) #因为query1和query2的是一样的

        self.train1 = train1
        self.train2 = train2
        self.query1 = query1
        self.query2= query2
        self.gallery1 = gallery1
        self.gallery2=gallery2

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, _ = self.get_imagedata_info(self.train1)
        # self.num_train2_pids, self.num_train2_imgs, self.num_train2_cams, _ = self.get_imagedata_info(self.train2)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, _ = self.get_imagedata_info(self.query1)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, _ = self.get_imagedata_info(self.gallery1)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir1):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir1))
        if not osp.exists(self.dataset_dir2):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir2))
        if not osp.exists(self.train1_dir):
            raise RuntimeError("'{}' is not available".format(self.train1_dir))
        if not osp.exists(self.train2_dir):
            raise RuntimeError("'{}' is not available".format(self.train2_dir))
        if not osp.exists(self.query1_dir):
            raise RuntimeError("'{}' is not available".format(self.query1_dir))
        if not osp.exists(self.query2_dir):
            raise RuntimeError("'{}' is not available".format(self.query2_dir))
        if not osp.exists(self.gallery1_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery1_dir))
        if not osp.exists(self.gallery2_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery2_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
 #      pattern = re.compile(r'([-\d]+)_c(\d)')
        pattern = re.compile(r'([-\d]+)_c([\d]+)')
 #      pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
            #这里构建了一个字典
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # print(pid, camid)
            if pid == -1: continue  # junk images are just ignored
          # assert 0 <= pid <= 1501  # pid == 0 means background
          # assert 1 <= camid <= 6
            assert 1 <= camid <= 2
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid,1))

        return dataset
