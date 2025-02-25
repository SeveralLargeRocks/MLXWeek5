import torch
import datasets
import whisper
import numpy as np


# Load LibriSpeech ASR dataset from Hugging Face
def load_dataset(split="test", config_name="clean"):
    """Load streaming LibriSpeech dataset"""
    dataset = datasets.load_dataset(
        "openslr/librispeech_asr",
        name=config_name,
        split=split,
        trust_remote_code=True,
        streaming=True
    )
    return dataset

class WhisperPreprocessor(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # Load tokenizer without loading the full model
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
    
    def __iter__(self):
        for sample in self.dataset:
            # Convert audio to float32 and preprocess
            audio_array = sample["audio"]["array"].astype(np.float32)
            audio_array = whisper.pad_or_trim(audio_array)
            
            # Tokenize text with SOT and EOT token
            tokenized_text = self.tokenizer.encode(sample["text"])
            tokenized_text = [self.tokenizer.sot] + tokenized_text
            tokenized_text = tokenized_text + [self.tokenizer.eot]
            
            # Return raw audio instead of mel spectrogram
            yield {
                "audio": audio_array,  # Raw audio waveform
                "tokens": tokenized_text
            }

def collate_fn(batch):
    """Collate function that returns tensor outputs"""
    # Process audio waveforms
    audios = torch.stack([torch.tensor(item["audio"]) for item in batch])
    
    # Process tokenized texts to handle variable lengths
    max_len = max(len(item["tokens"]) for item in batch)
    batch_size = len(batch)
    
    # Create tensor with padding
    tokens_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    # Fill tensor with token ids
    for i, item in enumerate(batch):
        tokens = item["tokens"]
        tokens_tensor[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    
    return {
        "audio": audios,        # Shape: [batch_size, samples]
        "tokens": tokens_tensor # Shape: [batch_size, seq_len]
    }

# Example usage
if __name__ == "__main__":
    dataset = load_dataset(split="test")
    whisper_dataset = WhisperPreprocessor(dataset)
    
    loader = torch.utils.data.DataLoader(
        whisper_dataset, 
        batch_size=4, 
        collate_fn=collate_fn
    )
    
    # Process one batch
    print("Loading batch...")
    batch = next(iter(loader))
    
    print(f"Audio waveforms: {batch['audio'].shape}, {batch['audio'].dtype}")
    print(f"Tokenized text: {batch['tokens'].shape}, {batch['tokens'].dtype}")
