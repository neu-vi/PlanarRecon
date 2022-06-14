import importlib
from .collate_fn import collate_fn


# find the dataset definition by name, for example ScanNetDataset (scannet.py)
def find_dataset_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    if dataset_name == 'scannet':
        return getattr(module, "ScanNetDataset")
    elif dataset_name == 'demo':
        return getattr(module, "DemoDataset")
