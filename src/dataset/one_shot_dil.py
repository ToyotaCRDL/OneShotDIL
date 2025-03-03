import numpy as np
import torch
import torchvision.transforms as transforms

from dataset.utils import *
from nets import get_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        new_data_transform = transforms.Compose(
            [transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(size=32, scale=(0.5, 0.95)),
            transforms.RandomAffine(degrees=[-20, 20], translate=(0.1, 0.1), scale=(0.5, 1.5)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3)]
        )
        
        dataset_train_org = CIFAR10(root=args.dataset_dir, train=True, transform=transform, download=True)
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
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        new_data_transform = transforms.Compose(
            [transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(size=28, scale=(0.5, 0.95)),
            transforms.RandomAffine(degrees=[-20, 20], translate=(0.1, 0.1), scale=(0.5, 1.5)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3)]
        )
        
        dataset_train_org = MNIST(root=args.dataset_dir, train=True, transform=transform, download=True)
        dataset_test = MNIST(root=args.dataset_dir, train=False, transform=transform, download=True)
    else:
        raise ValueError('Unknown dataset name')
    
    try:
        new_domain_class = transform_labels[args.new_domain]
        original_domain_class = transform_labels[args.original_domain_for_increment]
    except ValueError:
        raise ValueError('Miss match between dataset and domain class')
         
        
    if test:
        
        original_dataset_test = exclusion_class(dataset_test, new_domain_class)
        new_dataset_test = extraction_class(dataset_test, new_domain_class, original_domain_class)
        all_dataset_test = all_class(dataset_test, new_domain_class, original_domain_class)
        
        return original_dataset_test, new_dataset_test, all_dataset_test
    
    
    original_trainset_org = exclusion_class(dataset_train_org, new_domain_class)
    new_trainset_org = extraction_class(dataset_train_org, new_domain_class, original_domain_class)    

    indices = np.arange(len(original_trainset_org))
    id = int(0.8 * len(dataset_train_org) * len(original_trainset_org) / (len(original_trainset_org) + len(new_trainset_org)))
    original_dataset_train = torch.utils.data.Subset(original_trainset_org, indices[:id])
    original_dataset_val = torch.utils.data.Subset(original_trainset_org, indices[id:])

    indices = np.arange(len(new_trainset_org))
    id = int(0.8 * len(dataset_train_org) * len(new_trainset_org) / (len(original_trainset_org) + len(new_trainset_org)))
    new_dataset_train = torch.utils.data.Subset(new_trainset_org, indices[:id])

    new_trainloader = torch.utils.data.DataLoader(new_dataset_train, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    # Load a model trained on the original data
    net = get_net(args).to(device)

    ok_detect, ng_detect = split_dataset_by_prediction(new_trainloader, net)
    print("#OK (new domain train) =", len(ok_detect))
    print("#NG (new domain train) =", len(ng_detect))
    dataset_ng = torch.utils.data.Subset(new_dataset_train, ng_detect)
    

    perm = torch.randperm(len(dataset_ng))
    random_selected_idx = perm[0]
    new_data = torch.utils.data.Subset(dataset_ng, [random_selected_idx])
    
    
    perm = torch.randperm(len(original_dataset_train))
    sub_indexes = perm[:1000]
    sub_original_dataset = torch.utils.data.Subset(original_dataset_train, sub_indexes)

    return new_data, new_data_transform, sub_original_dataset, original_dataset_val