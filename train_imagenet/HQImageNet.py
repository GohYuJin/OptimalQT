import json
import os
import torch
import torchvision
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

class CustomImageNet(torchvision.datasets.VisionDataset):
    def __init__(self, root, impath_csv, imagenet_class_index_json, transforms=None, target_transform=None):
        super().__init__(root, transforms=transforms, target_transform=target_transform)
        self.im_paths = open(impath_csv, "r").read().strip().split("\n")
        id_classname_json = json.load(open(imagenet_class_index_json, "r"))
        self.label_mapping = {v[0]:int(k) for k, v in id_classname_json.items()}
        self.targets = [self.label_mapping[i.split("/")[1]] for i in self.im_paths]
        self.samples = list(zip(self.im_paths, self.targets))
        self.loader = torchvision.datasets.folder.default_loader
                
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
    
    def __len__(self):
        return len(self.im_paths)