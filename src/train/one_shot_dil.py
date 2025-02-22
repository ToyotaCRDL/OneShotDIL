import os
import copy
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from utils import compute_acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(net, image, label, new_dataset, new_data_transform, original_dataset, learning_rate, max_iters, delta, args):

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    original_trainloader = torch.utils.data.DataLoader(original_dataset, batch_size=args.batch_size-len(new_dataset), shuffle=True, num_workers=args.num_workers)

    count = 0
    while True:
        for i, (anchor_x, anchor_y) in enumerate(original_trainloader):
            augmented_x = []
            augmented_y = []
            for x, y in new_dataset:
                augmented_x.append(new_data_transform(x))
                augmented_y.append(y)

            target_x = torch.stack(augmented_x)
            target_y = torch.tensor(augmented_y)

            x = torch.cat([target_x, anchor_x])
            y = torch.cat([target_y, anchor_y])
            data_perm = torch.randperm(len(y))
            x = x[data_perm]
            y = y[data_perm]

            net.train()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x.to(device))
            loss = criterion(outputs, y.to(device))
            loss.backward()
            optimizer.step()
            count += 1
            
            
            print(f'| iter {count} | loss {loss.item():.4f} |')
            
            net.eval()
            if F.softmax(net(image.unsqueeze(0).to(device)), dim=1)[:, label] > delta:
                
                return count
            elif count >= max_iters:
                
                return count


def training_and_tuning(net0, new_data, new_data_transform, original_dataset, original_dataset_val, args):
    
    learning_rate_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            
    dataset_ng_dup_list = [new_data] * (args.new_batch_size + 1)
                
    image, label = dataset_ng_dup_list[args.new_batch_size][0]
    new_dataset = torch.utils.data.ConcatDataset(dataset_ng_dup_list[:args.new_batch_size])
    
    original_valloader = torch.utils.data.DataLoader(original_dataset_val, batch_size=100, shuffle=False, num_workers=args.num_workers)

    delta = 0.99
    max_iters = 100
    count_list = []
    val_arr = np.zeros(len(learning_rate_list))   
    for j, lr in enumerate(learning_rate_list):
        
        print('Learning rate:', lr)
        
        net = copy.deepcopy(net0)
        count = train(net, image, label, new_dataset, new_data_transform, original_dataset, lr, max_iters, delta, args)
        net.eval()    
        count_list.append(count)
        
        if count < max_iters:
            os.makedirs(args.models_dir, exist_ok=True)
            torch.save(net.state_dict(), args.models_dir + f'/model_lr={lr}.pth')
            val_arr[j]  = compute_acc(original_valloader, net)

    if torch.all(torch.tensor(count_list) == max_iters):
        print('All learning rates reached the maximum number of iterations before the target predicted probability exceeded delta.')
    else:
        best_i = np.argmax(val_arr).item()
        best_learning_rate = learning_rate_list[best_i]

        tuning_results = {'best idx': best_i, 'best learning rate': best_learning_rate}
        
        os.makedirs(args.results_dir, exist_ok=True)
        with open(args.results_dir + '/tuning_results.json','w') as f:
            json.dump(tuning_results, f)