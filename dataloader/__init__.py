from torch.utils.data import DataLoader

from .toy_sin import toy_sin_pair, toy_sin_a, toy_sin_b
from .tractable_pair import tractable_pair


def prepare_dataloader(name, train_batch, test_batch, num_workers, **kwargs):
    if name == 'toy_sin_pair':
        train_data, test_data = toy_sin_pair(), toy_sin_pair()
    elif name == 'toy_sin_a':
        train_data, test_data = toy_sin_a(), toy_sin_a()
    elif name == 'toy_sin_b':
        train_data, test_data = toy_sin_b(), toy_sin_b()
    elif name == 'tractable_pair':
        train_data, test_data = tractable_pair(10000), tractable_pair(10000)
    
    train_loader = DataLoader(train_data,
                              batch_size=train_batch,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_data,
                              batch_size=test_batch,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=False,
                              num_workers=num_workers)

    return train_loader, test_loader