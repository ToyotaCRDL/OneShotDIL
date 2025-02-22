import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(net, original_dataset, args):
    
    cudnn.benchmark = True
    
    trainloader = torch.utils.data.DataLoader(original_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    net.train()
    for epoch in range(200):
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        print(f'| epoch {epoch+1} | loss {train_loss/(batch_idx+1):.4f} |')
    
    os.makedirs(args.original_models_dir, exist_ok=True)
    torch.save(net.state_dict(), args.original_models_dir + '/model.pth')