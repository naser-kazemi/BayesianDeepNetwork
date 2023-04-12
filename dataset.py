import torch
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST


def getSets(filtered_class=None, removed_filter=None) -> (MNIST, MNIST):
    """
    :param filtered_class:
    :param removed_filter:
    :return: a torch dataset
    """

    transformer = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train = MNIST(root='./data', train=True, download=True, transform=transformer)
    test = MNIST(root='./data', train=False, download=True, transform=transformer)

    if filtered_class is not None:
        train_loader = DataLoader(train, batch_size=len(train))
        train_labels = next(iter(train_loader))[1].squeeze()

        test_loader = DataLoader(test, batch_size=len(test))
        test_loader = next(iter(test_loader))[1].squeeze()

        if removed_filter:
            train_indices = torch.nonzero(train_labels != filtered_class).squeeze()
            test_indices = torch.nonzero(test_loader != filtered_class).squeeze()
        else:
            train_indices = torch.nonzero(train_labels == filtered_class).squeeze()
            test_indices = torch.nonzero(test_loader == filtered_class).squeeze()

        train = torch.utils.data.Subset(train, train_indices)
        test = torch.utils.data.Subset(test, test_indices)

    return train, test


if __name__ == '__main__':
    train, test = getSets(filtered_class=3, removed_filter=False)

    test_loader = DataLoader(test, batch_size=len(test))
    images, labels = next(iter(test_loader))

    print(images.shape)
    print(torch.unique(labels.squeeze()))
