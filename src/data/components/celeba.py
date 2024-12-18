from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import CelebA
import numpy as np
import os.path as osp
import torch


class CelebADataset(Dataset):

    dataset_dir = 'celeba'

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()
        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.prepare_data()

    def prepare_data(self) -> None:
        trainset = CelebA(root=self.dataset_dir,
                        split='train',
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize((64, 64)),  # Resize images to 28x28
                            transforms.ToTensor()
                        ]), 
                        target_transform=None
        )

        testset = CelebA(root=self.dataset_dir, 
                        split='test',
                        download=True,
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
        return image, self.to_categorical(label_tensor, 40)
