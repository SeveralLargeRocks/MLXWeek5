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
import wandb

wandb.init(project='the-beast')

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

    encoder_output = None
    for epoch in range(100):
        total_loss = 0

        for i, (file, transcript) in enumerate(dataloader):
            file = file[0]
            transcript = transcript[0]

            waveform = whisper.load_audio(os.path.join(dirname, '../split', file))
            tokens = text_to_input_tks(transcript, tokenizer, device)

            if encoder_output is None:
                encoder_output = model.encode(waveform)

            # forward pass
            predictions = model(encoder_output, tokens)
            loss = get_loss(predictions, tokens, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item
            print(f"Step {i + 1}/{len(dataloader)}, Loss: {loss_item:.4f}")

            # # break after 3 batches for easy testing
            # if i > 3:
            #     break

            avg_loss = total_loss / (i + 1)

            wandb.log({
                "avg_train_loss": avg_loss,
                "loss_item": loss_item,
            })

        wandb.log({
            "epoch_loss": total_loss / len(dataloader),
            "epoch": epoch + 1
        })

        model_path = os.path.join(dirname, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 25 == 0:
            artifact = wandb.Artifact("the_beast", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
    
        print(f"Epoch {epoch} complete. Average loss: {total_loss / len(dataloader)}")
        # break after 1 epoch for easy testing

    # test on hello.wav
    model.eval()
    torch.set_grad_enabled(False)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    model = TwoTowerModel(hf_token)

    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = torch.nn.CrossEntropyLoss()

    print(f"Using device: {device}")

    train_model(model, tokenizer, optimizer, criterion, device)
