import torch
import whisper
from dataloader import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, valid_dataloader = get_dataloaders(batch_size=2)

model = whisper.load_model("tiny.en", device=device)
model.eval()

tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)

for audio, text in train_dataloader:
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device).unsqueeze(0)

    target_ids = tokenizer.encode(text)
    sot_token = torch.tensor(
        [tokenizer.sot], dtype=torch.long, device=device
    ).unsqueeze(0)
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_tks = torch.cat([sot_token, target_tensor], dim=-1)

    predictions = model(tokens=input_tks, mel=mel)




