"""Speech recognition using OpenAI's Whisper model."""

import torch
import tqdm
import wandb

from src.utils import (
    audio_path_to_mel,
    text_to_input_tks,
    get_loss,
    get_training_kit,
)
from src.dataloader import get_dataloaders

# Initialize wandb
wandb.init(project="whisper-finetuning", name="librispeech-training")

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
            train_dataloader, desc=f"Epoch {epoch + 1}"
        ):
            batch_step += 1
            audio = audio[0]
            text = text[0]
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

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
        })

    # Save the model after training
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Log model to wandb
    artifact = wandb.Artifact('whisper_model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    print("you silly sausage you're done")

    wandb.finish()


if __name__ == "__main__":
    device, model, tokenizer, optimizer, criterion = get_training_kit()

    print(f"Using device: {device}")

    train_model(model, tokenizer, optimizer, criterion, device)
