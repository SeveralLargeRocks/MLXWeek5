from datasets import load_dataset
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AdamW
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
import aiohttp
from tqdm import tqdm

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = load_dataset("librispeech_asr", "clean", split="train.100", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}, streaming=True)
valid_dataset = load_dataset("librispeech_asr", "clean", split="validation", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}, streaming=True)

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,  # Ensure this matches your dataset
    n_mels=80,          # Number of frequency bins
    n_fft=400,          # FFT window size
    hop_length=160,     # Hop length between frames
)

def collate_fn(batch):
    # Separate input_features and labels
    input_features = [item["input_features"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad input_features to the same length
    input_features = pad_sequence(input_features, batch_first=True, padding_value=0)

    # Pad labels to the same length
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignored labels

    return {"input_features": input_features, "labels": labels}

def preprocess_function(batch):
    # Load audio and compute mel spectrogram
    audio = batch["audio"]
    waveform = torch.tensor(audio["array"], dtype=torch.float16)
    sample_rate = audio["sampling_rate"]

    # Compute mel spectrogram
    mel = mel_transform(waveform)  # Shape: (n_mels, time_steps)
    mel = mel.squeeze(0)  # Remove batch dimension

    # Pad or truncate to 3000 time steps
    target_time_steps = 3000
    if mel.size(1) < target_time_steps:
        # Pad with zeros
        padding = torch.zeros((mel.size(0), target_time_steps - mel.size(1)))
        mel = torch.cat([mel, padding], dim=1)
    else:
        # Truncate to 3000 time steps
        mel = mel[:, :target_time_steps]

    # Transpose to (time_steps, n_mels)
    # Shape: (3000, 80)

    # Tokenize the transcription and convert to tensor
    labels = processor.tokenizer(batch["text"]).input_ids
    labels = torch.tensor(labels, dtype=torch.long)  # Convert to tensor
    return {"input_features": mel, "labels": labels}



train_dataset = train_dataset.map(preprocess_function, batched=False)
valid_dataset = valid_dataset.map(preprocess_function, batched=False)

train_dataloader = DataLoader(train_dataset, batch_size=20, collate_fn=collate_fn, num_workers=1)
valid_dataloader = DataLoader(valid_dataset, batch_size=20, collate_fn=collate_fn, num_workers=1)

scaler = GradScaler()
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss= []

for epoch in range(3):  # Number of epochs
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        # Move data to device
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()

        # Forward pass
        with autocast():
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss

        # Backward pass
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"Loss: {loss.item()}")
        
model.eval()
for batch in tqdm(valid_dataloader, desc="Validation"):
    input_features = batch["input_features"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        outputs = model(input_features=input_features, labels=labels)
        val_loss = outputs.loss

    print(f"Validation Loss: {val_loss.item()}")