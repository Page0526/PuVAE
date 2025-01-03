from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import MNIST
import numpy as np
import os.path as osp
import torch

class MnistDataset(Dataset):

    dataset_dir = 'mnist'

    def __init__(self, data_dir: str='data') -> None:
        super().__init__()
        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.prepare_data()
        
    def prepare_data(self) -> None:
        trainset = MNIST(root=self.dataset_dir,
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Resize((32, 32))  # Resize images to 28x28
                        ]), 
                        target_transform=None
        )

        testset = MNIST(root=self.dataset_dir, 
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Resize((32, 32))
                        ]),
                        target_transform=None
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
        return image, self.to_categorical(label_tensor, 10)

if __name__ == "__main__":
    dataset = MnistDataset(data_dir='data')
    print(len(dataset))
    image, cond = dataset[0]
    label = cond['label']
    print(image.shape, label)
    image.show()