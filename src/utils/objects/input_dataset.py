import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader


class InputDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index].input

    def get_loader(self, batch_size, shuffle=True, balance=False):
        if balance:
            targets = self.dataset.target.astype(int).to_numpy()
            class_counts = np.bincount(targets)
            class_counts[class_counts == 0] = 1
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            return DataLoader(dataset=self, batch_size=batch_size, sampler=sampler)

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)
