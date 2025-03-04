"""Speech recognition using OpenAI's Whisper model."""

import torch

from src.utils import (
    audio_path_to_mel,
    text_to_input_tks,
    get_loss,
    transcribe,
    get_training_kit,
)


def train_model(
    model,
    tokenizer,
    optimizer,
    criterion,
    device: str = "cpu",
    audio_path: str = "hello.wav",
) -> None:
    """Initialize and load Whisper model on specified device."""

    # process the audio file
    mel = audio_path_to_mel(audio_path, device)

    # Tokenize ground truth text
    input_tks = text_to_input_tks("Hello, my name is Izaak.", tokenizer, device)

    # Train the model
    model.train()
    for step in range(5):
        # Forward pass
        predictions = model(tokens=input_tks, mel=mel)
        loss = get_loss(predictions, input_tks, criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step + 1}/5, Loss: {loss.item():.4f}")

    # Test the model
    model.eval()
    torch.set_grad_enabled(False)
    result = model.transcribe(audio_path)
    print("Transcription: ", result["text"])


if __name__ == "__main__":
    device, model, tokenizer, optimizer, criterion = get_training_kit()

    print(f"Using device: {device}")

    model.eval()
    audio_path = "hello.wav"
    print("Transcription: ", transcribe(model, audio_path))

    train_model(model, tokenizer, optimizer, criterion, device, audio_path)
