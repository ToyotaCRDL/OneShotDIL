import torch


def get_net(args):
    
    if args.dataset_name == 'CIFAR10':
        if args.training_phase == 'only original domain':
            from nets.for_cifar.standard_resnet import ResNet18
            net = ResNet18(num_classes=9)
        elif args.training_phase == 'domain incremental learning':
            if args.fixed_stats:
                from nets.for_cifar.fixed_stats_resnet import ResNet18
            else:
                from nets.for_cifar.standard_resnet import ResNet18
            net = ResNet18(num_classes=9)
            try:
                net.load_state_dict(torch.load(args.original_models_dir + '/model.pth'))
            except FileNotFoundError:
                raise FileNotFoundError('You must train the original domain model before domain incremental learning.')
        elif args.training_phase == 'evaluation':
            from nets.for_cifar.standard_resnet import ResNet18
            net = ResNet18(num_classes=9)
        else:
            raise ValueError('Unknown training phase')
    elif args.dataset_name == 'MNIST':
        if args.training_phase == 'only original domain':
            from nets.for_mnist.standard_resnet import resnet18
            net = resnet18(num_classes=9, grayscale=True)
        elif args.training_phase == 'domain incremental learning':
            if args.fixed_stats:
                from nets.for_mnist.fixed_stats_resnet import resnet18
            else:
                from nets.for_mnist.standard_resnet import resnet18
            net = resnet18(num_classes=9, grayscale=True)
            try:
                net.load_state_dict(torch.load(args.original_models_dir + '/model.pth'))
            except FileNotFoundError:
                raise FileNotFoundError('You must train the original domain model before domain incremental learning.')
        elif args.training_phase == 'evaluation':
            from nets.for_mnist.standard_resnet import resnet18
            net = resnet18(num_classes=9, grayscale=True)
        else:
            raise ValueError('Unknown training phase')
    else:
        raise ValueError('Unknown dataset')
    
    return net