import copy
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def split_dataset_by_prediction(dataloader, model, return_ng_labels=False):
    ok = []
    ng = []
    ng_labels = []
    with torch.no_grad():
        model.eval()
        for batch_id, (images, labels) in enumerate(dataloader):
            inputs = images.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(inputs.shape[0]):
                img_id = batch_id * dataloader.batch_size + i
                if predicted[i] == labels[i]:
                    ok.append(img_id)
                else:
                    ng.append(img_id)
                    ng_labels.append(predicted[i])
    if return_ng_labels:
        return ok, ng, ng_labels
    else:
        return ok, ng


def exclusion_class(dataset, new_domain_class):
    exclusion_dataset = copy.deepcopy(dataset)

    exclusion_dataset.data = [dataset.data[idx] for idx, n in enumerate(dataset.targets) if n != new_domain_class]

    target_list = []
    for n in dataset.targets:
        if n < new_domain_class:
            target_list.append(n)
        elif n > new_domain_class:
            target_list.append(n-1)

    exclusion_dataset.targets = target_list
    
    return exclusion_dataset

def extraction_class(dataset, new_domain_class, original_domain_class):
    extraction_dataset = copy.deepcopy(dataset)
    extraction_dataset.data = [dataset.data[idx] for idx, n in enumerate(dataset.targets) if n == new_domain_class]
    if new_domain_class > original_domain_class:
        extraction_dataset.targets = [original_domain_class for n in dataset.targets if n == new_domain_class]
    else:
        extraction_dataset.targets = [original_domain_class - 1 for n in dataset.targets if n == new_domain_class]
    return extraction_dataset

def all_class(dataset, new_domain_class, original_domain_class):
    all_dataset = copy.deepcopy(dataset)

    target_list = []
    for n in dataset.targets:
        if n < new_domain_class:
            target_list.append(n)
        elif n == new_domain_class:
            if new_domain_class > original_domain_class:
                target_list.append(original_domain_class)
            else:
                target_list.append(original_domain_class - 1)
        elif n > new_domain_class:
            target_list.append(n-1)

    all_dataset.targets = target_list

    return all_dataset