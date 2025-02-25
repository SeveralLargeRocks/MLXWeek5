import aiohttp
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class LibriSpeechDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        file = item["file"]
        text = item["text"]
        return file, text


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


if __name__ == "__main__":
    batch_size = 2
    train_dataset, valid_dataset = load_librispeech()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    print("im done bro")
    files, items = next(iter(train_dataloader))
    print(files)
    print(items)
