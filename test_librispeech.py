import torch
from dataloader import get_dataloaders
from utils import audio_path_to_mel, text_to_input_tks, get_loss, get_training_kit
import tqdm
from jiwer import wer
from whisper.normalizers import BasicTextNormalizer

use_fine_tuned = False

normalizer = BasicTextNormalizer()

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, valid_dataloader = get_dataloaders(batch_size=1)

device, model, tokenizer, _, criterion = get_training_kit()

# Load the trained weights
if use_fine_tuned:
    model.load_state_dict(torch.load('trained_model_3.pth', map_location=device))
model.eval()

total_loss = 0
all_predictions = []
all_references = []

index = 0
for audio, text in tqdm.tqdm(valid_dataloader, desc="Testing"):
    index += 1
    audio = audio[0]
    text = text[0]
    mel = audio_path_to_mel(audio, device)
    input_tks = text_to_input_tks(text, tokenizer, device)
    with torch.no_grad():  # Add this for inference
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

print(f"Total loss: {total_loss:.4f}")
print(f"Average loss: {avg_loss:.4f}")
print(f"Word Error Rate: {wer_score:.2%}")

print("we done no cap fo real ma jigga")

"""
RESULTS:
Total loss: 4875.0953
Average loss: 1.8036
Word Error Rate: 34.67%
"""
