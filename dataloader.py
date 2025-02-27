import aiohttp
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import os


class LibriSpeechDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        file_path = item["file"]

        # Check if the file exists
        if not os.path.exists(file_path):
            # Try to find the file in the LibriSpeech directory structure
            file_name = os.path.basename(file_path)

            # Search in data directory for LibriSpeech files
            data_dir = "./data"
            for root, dirs, files in os.walk(data_dir):
                if file_name in files:
                    file_path = os.path.join(root, file_name)
                    break

        text = item["text"]
        return file_path, text


def load_librispeech():
    train_dataset = load_dataset(
        path="librispeech_asr",
        name="clean",
        split="train.100",
        cache_dir="./data",
        trust_remote_code=True,
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )
    valid_dataset = load_dataset(
        path="librispeech_asr",
        name="clean",
        split="validation",
        cache_dir="./data",
        trust_remote_code=True,
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )

    return LibriSpeechDataset(train_dataset), LibriSpeechDataset(valid_dataset)


def get_dataloaders(batch_size: int = 2):
    train_dataset, valid_dataset = load_librispeech()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    batch_size = 2
    train_dataloader, valid_dataloader = get_dataloaders(batch_size)

    print("im done bro")
    files, items = next(iter(train_dataloader))
    print(files)
    print(items)
