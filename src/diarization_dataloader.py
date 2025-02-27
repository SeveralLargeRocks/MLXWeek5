from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import os
import pandas

dirname = os.path.dirname(__file__)

class DiarizationDataset(Dataset):
    def __init__(self):
        self.dataset = pandas.read_csv(os.path.join(dirname, '../output.csv'), header=None, names=['file', 'transcript'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return row['file'], row['transcript']


def collate_fn(batch):
    return batch

if __name__ == "__main__":
    batch_size = 2

    dataset = DiarizationDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)

    print("im done bro")
    files, items = next(iter(dataloader))
    print(files)
    print(items)
