import argparse
import json
import torch

from utils import torch_fix_seed, compute_acc
from dataset import get_dataset
from nets import get_net


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='OneShotDIL')
    argparser.add_argument('--dataset_name', type=str, default='CIFAR10', help='Name of dataset')
    argparser.add_argument('--dataset_dir', type=str, default='./data', help='Directory to save dataset')
    argparser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    argparser.add_argument('--models_dir', type=str, default='./models', help='Directory to save dil models')
    argparser.add_argument('--original_models_dir', type=str, default='./original_models', help='Directory to save original models')
    argparser.add_argument('--seed', type=int, default=0, help='Seed')
    argparser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    argparser.add_argument('--training_phase', type=str, default='evaluation', help='Training phase')
    argparser.add_argument('--new_domain', type=str, default='truck', help='Class for new domain')
    argparser.add_argument('--original_domain_for_increment', type=str, default='automobile', help='Class for original domain to which new domain is added')
    args = argparser.parse_args()
    
    torch_fix_seed(args.seed)
    
    
    original_dataset_test, new_dataset_test, all_dataset_test = get_dataset(args, test=True)
    
    
    original_testloader = torch.utils.data.DataLoader(original_dataset_test, batch_size=100, shuffle=False, num_workers=args.num_workers)
    new_testloader = torch.utils.data.DataLoader(new_dataset_test, batch_size=100, shuffle=False, num_workers=args.num_workers)
    all_testloader = torch.utils.data.DataLoader(all_dataset_test, batch_size=100, shuffle=False, num_workers=args.num_workers)
    
    
    net = get_net(args).to(device)
    try:
        net.load_state_dict(torch.load(args.original_models_dir + '/model.pth'))
    except FileNotFoundError:
        raise FileNotFoundError('You must train the original domain model before evaluation.')
    
    original_test_acc = compute_acc(original_testloader, net)
    new_test_acc = compute_acc(new_testloader, net)
    all_test_acc = compute_acc(all_testloader, net)
    
    print('\t')
    print('Original domain model')
    print('---------------------')
    print(f'original test accuracy: {original_test_acc:.4f}')
    print(f'new test accuracy: {new_test_acc:.4f}')
    print(f'all test accuracy: {all_test_acc:.4f}')
    print('---------------------')
    
    try:
        with open(args.results_dir + '/tuning_results.json', 'r') as f:
            tuning_results = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('You must perform domain incremental learning before evaluation.')   
    
    lr = tuning_results['best learning rate']
    
    net.load_state_dict(torch.load(args.models_dir + f'/model_lr={lr}.pth'))
    
    
    original_test_acc = compute_acc(original_testloader, net)
    new_test_acc = compute_acc(new_testloader, net)
    all_test_acc = compute_acc(all_testloader, net)
    
    print('\t')
    print('One-shot DIL model')
    print('---------------------')
    print(f'original test accuracy: {original_test_acc:.4f}')
    print(f'new test accuracy: {new_test_acc:.4f}')
    print(f'all test accuracy: {all_test_acc:.4f}')
    print('---------------------')
    