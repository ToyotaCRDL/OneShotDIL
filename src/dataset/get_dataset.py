


def get_dataset(args, test=False):
    
    if args.training_phase == 'only original domain':
        from dataset.standard import make_dataset
    elif args.training_phase == 'domain incremental learning':
        from dataset.one_shot_dil import make_dataset
    elif args.training_phase == 'evaluation':
        from dataset.one_shot_dil import make_dataset
    else:
        raise ValueError('Unknown training phase')
    
    return make_dataset(args, test)