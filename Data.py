import torch
from torchvision import datasets, transforms

import json
from PIL import Image
from random import choice

class Data:
    
    def __init__(self, root_directory):
        
        # input data directory
        self._root_dir = root_directory
        
        # data transforms
        self._train_transforms = transforms.Compose([transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                     transforms.RandomRotation(degrees=(-180, 180)),
                                                     transforms.ColorJitter(brightness=0.5, contrast=1.0, saturation=0.1, hue=0.1),
                                                     transforms.RandomHorizontalFlip(p=0.333),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])])
        self._validation_transforms = transforms.Compose([transforms.Resize(256),
                                                          transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                               [0.229, 0.224, 0.225])])
        self._test_transforms = self._validation_transforms
        
        # data sets
        self._train_data = datasets.ImageFolder(self._root_dir + r"/train", transform=self._train_transforms)
        self._validation_data = datasets.ImageFolder(self._root_dir + r"/valid", transform=self._validation_transforms)
        self._test_data = datasets.ImageFolder(self._root_dir + r"/test", transform=self._test_transforms)
        
        # label hash mapping
        self._index_to_label = {v : k for k, v in self._test_data.class_to_idx.items()}
        with open('cat_to_name.json', 'r') as f:
            self._label_to_name = json.load(f)
        
        # data loaders
        self.trainloader = torch.utils.data.DataLoader(self._train_data, batch_size=32)
        self.testloader = torch.utils.data.DataLoader(self._test_data, batch_size=32)
        self.valloader= torch.utils.data.DataLoader(self._validation_data, batch_size=32)
        
        # helper variables
        self.n_outputs = len(self._train_data.class_to_idx.items())
    
    
    def set_index_to_label(self, class_to_idx):
        self._index_to_label = {v : k for k, v in class_to_idx.items()}
    
       
    def get_name(self, label=None, index=None):
        """ return flower name from label or index """
        
        if label:
            return self._label_to_name[label]
        elif index:
            return self._label_to_name[self._index_to_label[index]]
        else:
            raise KeyError("No label or index key provided!")
    
    
    def get_random_test_image(self):
        image_tensor, image_index = choice(self._test_data)
        image_tensor.unsqueeze_(0)        
        return image_tensor, image_index
    
    
    def get_image_from_path(self, path):
        image_tensor = self._test_transforms(Image.open(path)).unsqueeze(0)
        image_index = self._test_data.class_to_idx[path.split('/')[-2]]
        return image_tensor, image_index