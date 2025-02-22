import argparse
import torch

from utils import torch_fix_seed
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
    argparser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    argparser.add_argument('--seed', type=int, default=0, help='Seed')
    argparser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    argparser.add_argument('--training_phase', type=str, default='only original domain', help='Training phase')
    argparser.add_argument('--new_domain', type=str, default='truck', help='Class for new domain')
    argparser.add_argument('--original_domain_for_increment', type=str, default='automobile', help='Class for original domain to which new domain is added')
    argparser.add_argument('--new_batch_size', type=int, default=32, help='Size of the mini-batch sampled by duplicating new data')
    argparser.add_argument('--fixed_stats', action='store_true', help='Use fixed stats (proposed method)')
    args = argparser.parse_args()
    
    
    torch_fix_seed(args.seed)
    
    dataset = get_dataset(args)
    
    net = get_net(args).to(device)
    
    if args.training_phase == 'only original domain':
        from train.standard import train
        train(net, dataset, args)
    elif args.training_phase == 'domain incremental learning':        
        from train.one_shot_dil import training_and_tuning
        training_and_tuning(net, *dataset, args)
    else:
        raise ValueError('Unknown training phase')
    
    
    
    