import random
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_acc(dataloader, model):

    total = 0
    success = 0
    with torch.no_grad():
        model.eval()
        for images, labels in dataloader:
            total += images.shape[0]
            inputs = images.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            success += torch.sum(predicted == labels).item()
    return success / total

def torch_fix_seed(seed):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True