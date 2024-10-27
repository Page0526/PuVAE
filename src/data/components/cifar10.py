import numpy as np
import os.path as osp
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import CIFAR10


class Cifar10Dataset(Dataset):

    dataset_dir = 'cifar10'

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.prepare_data()

    def prepare_data(self) -> None:
        trainset = CIFAR10(self.dataset_dir, 
                           download=True, 
                           train=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),   # Convert NumPy arrays to PyTorch tensors
                               transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
                               ]), # images come as PIL format, we want to turn into Torch tensors
                               target_transform=None # you can transform labels as well)
                        )
        testset = CIFAR10(self.dataset_dir, # where to download data to?
                          train=False, # get training data
                          download=True, # download data if it doesn't exist on disk
                          transform=transforms.Compose([
                              transforms.ToTensor(),   # Convert NumPy arrays to PyTorch tensors
                              transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
                              ]), # images come as PIL format, we want to turn into Torch tensors
                              target_transform=None # you can transform labels as well
        )

        self.dataset = ConcatDataset(datasets=[trainset, testset])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return np.array(self.dataset[index][0]), {
            'label': self.dataset[index][1]
        }


if __name__ == "__main__":
    dataset = Cifar10Dataset(data_dir='data')
    print(len(dataset))
    image, cond = dataset[0]
    label = cond['label']
    print(image.shape, label)
    image.show()