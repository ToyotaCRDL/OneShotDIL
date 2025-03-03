import numpy as np
import torchvision.transforms as transforms

from dataset.utils import *


def make_dataset(args, test=False):
    
    if args.dataset_name == 'CIFAR10':
        
        from torchvision.datasets import CIFAR10
        
        transform_labels = {
            "airplane":   0,
            "automobile": 1,
            "bird":       2,
            "cat":        3,
            "deer":       4,
            "dog":        5,
            "frog":       6,
            "horse":      7,
            "ship":       8,
            "truck":      9
        }
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        dataset_train_org = CIFAR10(root=args.dataset_dir, train=True, transform=train_transform, download=True)
        dataset_test = CIFAR10(root=args.dataset_dir, train=False, transform=transform, download=True)
    elif args.dataset_name == 'MNIST':
        
        from torchvision.datasets import MNIST
        
        transform_labels = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9
        }
        
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset_train_org = MNIST(root=args.dataset_dir, train=True, transform=transform, download=True)
        dataset_test = MNIST(root=args.dataset_dir, train=False, transform=transform, download=True)
    else:
        raise ValueError('Unknown dataset name')    
    
    try:
        new_domain_class = transform_labels[args.new_domain]
        original_domain_class = transform_labels[args.original_domain_for_increment]
    except ValueError:
        raise ValueError('Miss match between dataset and new domain class')

    if test:
        testset = exclusion_class(dataset_test, new_domain_class)
        return testset
        
    original_trainset_org = exclusion_class(dataset_train_org, new_domain_class)
    new_trainset_org = extraction_class(dataset_train_org, new_domain_class, original_domain_class)
    
    indices = np.arange(len(original_trainset_org))
    id = int(0.8 * len(dataset_train_org) * len(original_trainset_org) / (len(original_trainset_org) + len(new_trainset_org)))
    original_dataset_train = torch.utils.data.Subset(original_trainset_org, indices[:id])
    
    
    return original_dataset_train