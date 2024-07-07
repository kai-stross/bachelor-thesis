from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym.register import register_dataset, register_loader
import torch
import os

class CustomDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        file_path = os.path.join(self.root, 'processed', f'{self.name}.pt')
        # Loading the Data object directly
        self._data = torch.load(file_path)

    @property
    def raw_file_names(self):
        # No raw files to process
        return []

    @property
    def processed_file_names(self):
        # List the file names that are expected in the processed directory
        return [f'{self.name}.pt']

    def len(self):
        # Returns the number of examples (graphs) in the dataset
        return 1  # Since we have one graph

    def get(self, idx):
        # Retrieves a single graph for use in a PyTorch DataLoader
        if idx == 0:
            return self._data
        else:
            raise IndexError(f'Index {idx} out of range for dataset with only one graph.')

    def download(self):
        # No download needed, as data is presumed to be pre-processed and available
        pass

    def process(self):
        # No processing needed since the data is already in the correct format
        pass

# Registering the loader function
@register_loader('chrome-run-01-without-centrality-metrics')
def load_my_dataset(format, name, dataset_dir):
    return CustomDataset(root=dataset_dir, name=name)