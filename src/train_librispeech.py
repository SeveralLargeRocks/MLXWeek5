"""Speech recognition using OpenAI's Whisper model."""

import torch
import tqdm
import wandb
from whisper.normalizers import BasicTextNormalizer
from src.utils import (
    audio_path_to_mel,
    text_to_input_tks,
    get_loss,
    get_training_kit,
)
from src.dataloader import get_dataloaders
from jiwer import wer

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
        for audio, text in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
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
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
            }
        )

    # Save the model after training
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)

    # Log model to wandb
    artifact = wandb.Artifact("whisper_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    print("you silly sausage you're done")

    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    normalizer = BasicTextNormalizer()

    index = 0
    for audio, text in tqdm.tqdm(valid_dataloader, desc="Testing"):
        index += 1
        audio = audio[0]
        text = text[0]
        mel = audio_path_to_mel(audio, device)
        input_tks = text_to_input_tks(text, tokenizer, device)
        predictions = model(tokens=input_tks, mel=mel)
        loss = get_loss(predictions, input_tks, criterion)
        total_loss += loss.item()

        # Decode predictions
        predicted_token_ids = torch.argmax(predictions, dim=-1)
        # Include the first token by starting from index 1 (after start token)
        pred_text = tokenizer.decode(predicted_token_ids[0])

        # Clean and normalize
        all_predictions.append(normalizer(pred_text))
        all_references.append(normalizer(text))

    avg_loss = total_loss / len(valid_dataloader)
    wer_score = wer(all_references, all_predictions)

    # Log evaluation metrics
    wandb.log(
        {
            "total_loss": total_loss,
            "eval_loss": avg_loss,
            "word_error_rate": wer_score,
            # Log a few example predictions
            "examples": wandb.Table(
                columns=["Reference", "Prediction"],
                data=list(zip(all_references[:5], all_predictions[:5])),
            ),
        }
    )

    print(f"Total loss: {total_loss:.4f}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Word Error Rate: {wer_score:.2%}")

    print("we done no cap fo real ma jigga")

    wandb.finish()


if __name__ == "__main__":
    device, model, tokenizer, optimizer, criterion = get_training_kit()

    print(f"Using device: {device}")

    train_model(model, tokenizer, optimizer, criterion, device)


"""

RESULTS:
Average loss: 0.0030
Word Error Rate: 28.89%
"""
