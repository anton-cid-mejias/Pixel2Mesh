import json
import os
import pickle

import numpy as np
import torch
from skimage import io, transform
from torch.utils.data.dataloader import default_collate
import pandas as pd
import csv

import config
from datasets.base_dataset import BaseDataset


class ShapeNet(BaseDataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(self, file_root, mesh_pos, normalization, options):
        super().__init__()
        self.file_root = file_root

        meta_path = os.path.join(self.file_root, "figures", "classes.csv")
        classes_dt = pd.read_csv(meta_path, delimiter=',')
        self.classes = np.squeeze(classes_dt[['classname']].values, axis=1).tolist()
        self.class_files = np.squeeze(classes_dt[['filename']].values, axis=1).tolist()

        self.file_names = []
        for file in self.class_files:
            file_path = os.path.join(self.file_root, "figures", file)
            files = pd.read_csv(file_path, delimiter=',')
            self.file_names.extend(np.squeeze(files.values, axis=1).tolist())

        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = options.resize_with_constant_border

    def __getitem__(self, index):
        filename = self.file_names[index]
        label = filename.split('_')[0]
        with open(filename.replace('.png', '.xyz'), 'r') as f:
            reader = csv.reader(f, dialect='excel')
            points = []
            normals = []
            for row in reader:
                row = row.split(' ')
                points.append(row[:3])
                normals.append(row[3:])
        img = io.imread(filename)
        img[np.where(img[:, :, 3] == 0)] = 255
        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)  # to match behavior of old versions
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)

        points -= np.array(self.mesh_pos)
        assert points.shape[0] == normals.shape[0]
        length = points.shape[0]

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "points": points,
            "normals": normals,
            "labels": [label],
            "filename": filename,
            "length": length
        }

    def __len__(self):
        return len(self.file_names)

def get_shapenet_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """
    def shapenet_collate(batch):
        if len(batch) > 1:
            all_equal = True
            for t in batch:
                if t["length"] != batch[0]["length"]:
                    all_equal = False
                    break
            points_orig, normals_orig = [], []
            if not all_equal:
                for t in batch:
                    pts, normal = t["points"], t["normals"]
                    length = pts.shape[0]
                    choices = np.resize(np.random.permutation(length), num_points)
                    t["points"], t["normals"] = pts[choices], normal[choices]
                    points_orig.append(torch.from_numpy(pts))
                    normals_orig.append(torch.from_numpy(normal))
                ret = default_collate(batch)
                ret["points_orig"] = points_orig
                ret["normals_orig"] = normals_orig
                return ret
        ret = default_collate(batch)
        ret["points_orig"] = ret["points"]
        ret["normals_orig"] = ret["normals"]
        return ret

    return shapenet_collate