import torch
from src.dataloader import get_dataloaders
from src.utils import audio_path_to_mel, text_to_input_tks, get_loss, get_training_kit

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, valid_dataloader = get_dataloaders(batch_size=1)

device, model, tokenizer, optimizer, criterion = get_training_kit()

model.eval()

total_loss = 0

for audio, text in valid_dataloader:
    audio = audio[0]
    text = text[0]
    mel = audio_path_to_mel(audio, device)
    input_tks = text_to_input_tks(text, tokenizer, device)
    predictions = model(tokens=input_tks, mel=mel)
    loss = get_loss(predictions, input_tks, criterion)
    total_loss += loss.item()


print(f"Total loss: {total_loss / len(train_dataloader)}")

print('we done no cap fo real ma jigga')