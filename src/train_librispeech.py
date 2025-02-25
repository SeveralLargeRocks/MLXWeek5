"""Speech recognition using OpenAI's Whisper model."""

import torch
import tqdm

from src.utils import (
    audio_path_to_mel,
    text_to_input_tks,
    get_loss,
    get_training_kit,
    get_dataloaders,
)


def train_model(
    model,
    tokenizer,
    optimizer,
    criterion,
    device: str = "cpu",
) -> None:
    """Initialize and load Whisper model on specified device."""

    train_dataloader, valid_dataloader = get_dataloaders(batch_size=1)

    # Train the model
    model.train()
    for epoch in range(5):
        batch_step = 0
        total_loss = 0
        for audio, text in tqdm.tqdm(
            train_dataloader, desc=f"training epoch {epoch + 1}"
        ):
            batch_step += 1

            mel = audio_path_to_mel(audio, device)
            input_tks = text_to_input_tks(text, tokenizer, device)

            predictions = model(tokens=input_tks, mel=mel)
            loss = get_loss(predictions, input_tks, criterion)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Step {batch_step + 1}/5, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Average loss: {avg_loss:.4f}")

    # Save the model after training
    torch.save(model.state_dict(), "trained_model.pth")

    print("you silly sausage you're done")


if __name__ == "__main__":
    device, model, tokenizer, optimizer, criterion = get_training_kit()

    print(f"Using device: {device}")

    train_model(model, tokenizer, optimizer, criterion, device)
