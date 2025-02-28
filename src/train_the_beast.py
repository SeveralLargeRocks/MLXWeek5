"""Speech recognition using OpenAI's Whisper model."""

import torch

from src.utils import (
    text_batch_to_input_tks,
    get_loss,
    get_training_kit,
    set_seed,
)
from src.diarization_dataloader import DiarizationDataset
from torch.utils.data import DataLoader
from src.model import TwoTowerModel
from dotenv import load_dotenv
import os
import whisper
import wandb
import numpy as np
import time

dirname = os.path.dirname(__file__)

def collate_fn(batch, max_len):
    print('-------------------- in collate fn ------------------')
    files, transcripts = zip(*batch)
    tokenized_texts = text_batch_to_input_tks(transcripts, model.tokenizer, device)

    # trim first
    tokenized_texts = [tokens[:max_len] for tokens in tokenized_texts]

    # then pad
    eot_token = model.tokenizer.eot  # End-of-transcript token
    padded_tokens = torch.stack([
        torch.cat([
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(eot_token, dtype=torch.long).repeat(max_len - len(tokens))
        ], dim=0)
        for tokens in tokenized_texts
    ])
    print('padded tokens shape', padded_tokens.shape)

    return files, padded_tokens

def train_model(
    model: TwoTowerModel,
    tokenizer,
    optimizer,
    criterion,
    device: str = "cpu",
) -> None:
    """Initialize and load Whisper model on specified device."""

    model.to(device)

    batch_size = 5
    dataset = DiarizationDataset()
    dataloader = DataLoader(dataset, batch_size, collate_fn=lambda batch: collate_fn(batch, model.max_len))

    # Train the model
    model.train()

    for epoch in range(5):
        total_loss = 0
        for i, (files, token_ids_batch) in enumerate(dataloader):

            print(f"converting {len(files)} to waveform")
            start_time = time.time()
            waveforms = np.array([whisper.load_audio(os.path.join(dirname, '../split', file)) for file in files])
            print('waveforms.shape: ', waveforms.shape)
            print(f"took {time.time() - start_time}")

            # tokens = text_batch_to_input_tks(transcripts, tokenizer, device)
            print('token_ids_batch.shape: ', token_ids_batch.shape)
            encoder_output = model.encode(waveforms)
            # print('encoder_outputs.shape: ', [x.shape for x in encoder_output])

            # forward pass
            predictions = model(encoder_output, token_ids_batch)
            print('predictions.shape: ', predictions.shape)
            loss = get_loss(predictions, token_ids_batch, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item
            print(f"Step {i + 1}/{len(dataloader)}, Loss: {loss_item:.4f}")

            # # break after 3 batches for easy testing
            # if i > 2:
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

        artifact = wandb.Artifact("the_beast", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
        print(f"Epoch {epoch} complete. Average loss: {total_loss / i}")
        # break after 1 epoch for easy testing
        break

    # test on hello.wav
    model.eval()
    torch.set_grad_enabled(False)


if __name__ == "__main__":
    set_seed(16)

    wandb.init(project='the-beast')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    model = TwoTowerModel(hf_token)

    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = torch.nn.CrossEntropyLoss()

    print(f"Using device: {device}")

    train_model(model, tokenizer, optimizer, criterion, device)
