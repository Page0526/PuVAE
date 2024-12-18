from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import LSUN
import numpy as np
import os.path as osp
import torch


class LSUNDataset(Dataset):

    dataset_dir = 'lsun'

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()
        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.prepare_data()

    def prepare_data(self) -> None:
        trainset = LSUN(root=self.dataset_dir,
                        classes=['bedroom_train'],
                        # download=True,
                        transform=transforms.Compose([
                            transforms.Resize((64, 64)),  # Resize images to 28x28
                            transforms.ToTensor()
                        ]), 
                        target_transform=None
        )

        testset = LSUN(root=self.dataset_dir, 
                        classes=['bedroom_test'],
                        # download=True,
                        transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor()
                        ]),
                        target_transform=None
        )
        self.dataset = ConcatDataset(datasets=[trainset, testset])

    def __len__(self):
        return len(self.dataset)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        label_tensor = torch.tensor(label, dtype=torch.long)
  
        image = np.array(image)
        return image, self.to_categorical(label_tensor, 10)
