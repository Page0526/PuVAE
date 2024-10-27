from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import FashionMNIST
import numpy as np
import os.path as osp
import torch


class FashionDataset(Dataset):

    dataset_dir = 'fashion'

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()
        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.prepare_data()

    def prepare_data(self):
        trainset = FashionMNIST(root=self.dataset_dir, # where to download data to?
                        train=True, # get training data
                        download=True, # download data if it doesn't exist on disk
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),  # Resize images if necessary
                            transforms.ToTensor(),  # Convert NumPy arrays to PyTorch tensors
                            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
                        ]), # images come as PIL format, we want to turn into Torch tensors
                        target_transform=None # you can transform labels as well
        )

        testset = FashionMNIST(root=self.dataset_dir, # where to download data to?
                        train=False, # get training data
                        download=True, # download data if it doesn't exist on disk
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),  # Resize images if necessary
                            transforms.ToTensor(),  # Convert NumPy arrays to PyTorch tensors
                            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
                        ]), # images come as PIL format, we want to turn into Torch tensors
                        target_transform=None # you can transform labels as well
        )

        self.dataset = ConcatDataset(datasets=[trainset, testset])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label_tensor = torch.tensor(label, dtype=torch.long)
        return np.array(image), {'label': label_tensor}


if __name__ == "__main__":
    dataset = FashionDataset(data_dir='data')
    print(len(dataset))
    image, cond = dataset[0]
    label = cond['label']
    print(image.shape, label)
    image.show()