"""Speech recognition using OpenAI's Whisper model."""

import torch

from src.utils import (
    text_to_input_tks,
    get_loss,
    get_training_kit,
)
from src.diarization_dataloader import DiarizationDataset
from torch.utils.data import DataLoader
from src.model import TwoTowerModel
from dotenv import load_dotenv
import os
import whisper

dirname = os.path.dirname(__file__)

torch.manual_seed(16)

def train_model(
    model,
    tokenizer,
    optimizer,
    criterion,
    device: str = "cpu",
) -> None:
    """Initialize and load Whisper model on specified device."""

    model.to(device)

    batch_size = 1
    dataset = DiarizationDataset()
    dataloader = DataLoader(dataset, batch_size)

    # Train the model
    model.train()

    for epoch in range(5):
        total_loss = 0
        for i, (file, transcript) in enumerate(dataloader):
            file = file[0]
            transcript = transcript[0]

            waveform = whisper.load_audio(os.path.join(dirname, '../split', file))
            tokens = text_to_input_tks(transcript, tokenizer, device)

            # forward pass
            predictions = model(waveform, tokens)
            loss = get_loss(predictions, tokens, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Step {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

            # break after 3 batches for easy testing
            if i > 3:
                break
        
        print(f"Epoch {epoch} complete. Average loss: {total_loss / i}")
        # break after 1 epoch for easy testing
        break

    # test on hello.wav
    model.eval()
    torch.set_grad_enabled(False)


if __name__ == "__main__":
    device, _, tokenizer, optimizer, criterion = get_training_kit()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    model = TwoTowerModel(hf_token)

    print(f"Using device: {device}")

    train_model(model, tokenizer, optimizer, criterion, device)
