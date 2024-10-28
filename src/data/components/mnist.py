from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import MNIST
import numpy as np
import os.path as osp
import torch

class MnistDataset(Dataset):

    dataset_dir = 'mnist'
    
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    def __init__(self, data_dir: str='data') -> None:
        super().__init__()
        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.prepare_data()
        
    def prepare_data(self) -> None:
        trainset = MNIST(root=self.dataset_dir, # where to download data to?
                        train=True, # get training data
                        download=True, # download data if it doesn't exist on disk
                        transform=transforms.Compose([
                            transforms.ToTensor(),   # Convert NumPy arrays to PyTorch tensors
                            # transforms.Resize((32, 32))  # Resize images to 28x28
                        ]), # images come as PIL format, we want to turn into Torch tensors
                        target_transform=None # you can transform labels as well
        )

        testset = MNIST(root=self.dataset_dir, # where to download data to?
                        train=False, # get training data
                        download=True, # download data if it doesn't exist on disk
                        transform=transforms.Compose([
                            transforms.ToTensor(),   # Convert NumPy arrays to PyTorch tensors, normalized to [0, 1]
                            # transforms.Resize((32, 32))  # Resize images to 28x28
                        ]), # images come as PIL format, we want to turn into Torch tensors
                        target_transform=None # you can transform labels as well
        )
        self.dataset = ConcatDataset(datasets=[trainset, testset])

    def __len__(self) -> int:
        return len(self.dataset)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label_tensor = torch.tensor(label, dtype=torch.long)
  
        image = np.array(image)
        # image = np.transpose(image, (1, 2, 0))
        return image, self.to_categorical(label_tensor, 10)

if __name__ == "__main__":
    dataset = MnistDataset(data_dir='data')
    print(len(dataset))
    image, cond = dataset[0]
    label = cond['label']
    print(image.shape, label)
    image.show()