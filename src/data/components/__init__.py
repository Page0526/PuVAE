from .cifar10 import Cifar10Dataset
from .mnist import MnistDataset
from .fashion import FashionDataset
# from .imagenet import ImageNetDataset


__datasets = {
    'cifar10': Cifar10Dataset,
    'mnist': MnistDataset,
    'fashion': FashionDataset,
    # 'imagenet': ImageNetDataset
}

def init_dataset(name, **kwargs):
    '''
    Initialize a dataset
    '''
    available_datasets = list(__datasets.keys())

    if name not in available_datasets:
        raise ValueError('Invald valid dataset name. Received "{}", but expected to be one of {}'.format(name, available_datasets))

    return __datasets[name](**kwargs)