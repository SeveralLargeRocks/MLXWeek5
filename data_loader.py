import torch
import datasets
import numpy as np
from typing import Dict, List, Any
import whisper


# Load LibriSpeech ASR dataset from Hugging Face
def load_librispeech_dataset(split="train.100", config_name="clean"):
    dataset = datasets.load_dataset(
        "openslr/librispeech_asr",
        name=config_name,  # 'clean' or 'other' for different quality levels
        split=split,       # e.g. 'train.clean.100', 'validation', 'test'
        trust_remote_code=True,
        streaming=True
    )
    return dataset

# Memory-efficient preprocessing wrapper
class AudioPreprocessor(torch.utils.data.IterableDataset):
    def __init__(self, dataset, max_audio_length=480000, whisper_processor=None):
        self.dataset = dataset
        self.max_audio_length = max_audio_length  # ~30 seconds at 16kHz
        self.whisper_processor = whisper_processor
    
    def __iter__(self):
        for sample in self.dataset:
            # Process only when needed, not storing everything in memory
            audio_array = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]
            
            # Apply preprocessing immediately and discard original audio
            if self.whisper_processor:
                # Use Whisper's preprocessing
                audio_array = whisper.pad_or_trim(audio_array)
                mel = whisper.log_mel_spectrogram(audio_array).numpy()
                
                yield {
                    "mel": mel,
                    "text": sample["text"],
                    "id": sample["id"],
                }
            else:
                # Basic preprocessing
                if len(audio_array) > self.max_audio_length:
                    audio_array = audio_array[:self.max_audio_length]
                
                yield {
                    "audio": audio_array,
                    "sample_rate": sample_rate,
                    "text": sample["text"],
                    "id": sample["id"],
                }

# Memory-efficient collate function
def efficient_collate_fn(batch):
    """Collate function that minimizes memory usage."""
    
    # For whisper-preprocessed data
    if "mel" in batch[0]:
        # Already preprocessed mels can be stacked directly
        mels = torch.stack([torch.tensor(item["mel"]) for item in batch])
        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]
        
        return {
            "mel": mels,
            "text": texts,
            "id": ids
        }
    
    # For raw audio data
    audio_lengths = [len(item["audio"]) for item in batch]
    max_len = max(audio_lengths)
    
    # Prepare arrays at the right size from the start
    batch_size = len(batch)
    audio_tensor = torch.zeros(batch_size, max_len)
    attention_mask = torch.zeros(batch_size, max_len)
    
    # Fill arrays efficiently
    for i, (item, length) in enumerate(zip(batch, audio_lengths)):
        # Copy directly to the pre-allocated tensor
        audio_tensor[i, :length] = torch.tensor(item["audio"], dtype=torch.float)
        attention_mask[i, :length] = 1.0
    
    return {
        "audio": audio_tensor,
        "attention_mask": attention_mask,
        "text": [item["text"] for item in batch],
        "sample_rate": batch[0]["sample_rate"],  # Assuming consistent sample rate
        "id": [item["id"] for item in batch],
    }

def create_memory_efficient_dataloader(
    split="test", 
    batch_size=4, 
    whisper_format=False,
    max_audio_length=480000,
):
    """Creates a memory-efficient dataloader for audio datasets."""
    # Load the streaming dataset
    dataset = load_librispeech_dataset(split=split)
    
    # Create processor based on desired output format
    whisper_processor = whisper.load_model("tiny", device="cpu") if whisper_format else None
    
    # Wrap with memory-efficient preprocessor
    efficient_dataset = AudioPreprocessor(
        dataset, 
        max_audio_length=max_audio_length,
        whisper_processor=whisper_processor if whisper_format else None
    )
    
    # Create dataloader with prefetch_factor to control memory usage
    return torch.utils.data.DataLoader(
        efficient_dataset,
        batch_size=batch_size,
        collate_fn=efficient_collate_fn,
        num_workers=2,  # Adjust based on your system
        prefetch_factor=2,  # Lower prefetch to reduce memory
        pin_memory=torch.cuda.is_available(),  # For faster GPU transfer
    )

# Example usage
if __name__ == "__main__":
    # Create memory-efficient dataloader
    test_loader = create_memory_efficient_dataloader(
        split="test", 
        batch_size=4,
        whisper_format=False  # Set to True for Whisper-compatible processing
    )
    
    # Process one batch to demonstrate
    print("Loading first batch...")
    batch = next(iter(test_loader))
    
    print(f"Batch audio shape: {batch['audio'].shape}")
    print(f"Sample rate: {batch['sample_rate']}")
    print(f"Text example: {batch['text'][0]}")
    
    # Display memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")