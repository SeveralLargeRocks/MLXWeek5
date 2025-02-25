"""Speech recognition using OpenAI's Whisper model."""

import torch
import whisper


def audio_path_to_mel(audio_path: str, device: str = "cpu") -> torch.Tensor:
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device).unsqueeze(0)
    return mel


def main(device: str = "cpu") -> None:
    """Initialize and load Whisper model on specified device."""

    model = whisper.load_model("tiny.en", device=device)
    model.eval()

    audio_path = "hello.wav"
    result = model.transcribe(audio_path)
    print("Transcription: ", result["text"])

    # process the audio file
    mel = audio_path_to_mel(audio_path, device)

    # Tokenize ground truth text
    ground_truth_text = "Hello, my name is Izaak."
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
    target_ids = tokenizer.encode(ground_truth_text)
    sot_token = torch.tensor(
        [tokenizer.sot], dtype=torch.long, device=device
    ).unsqueeze(0)
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(
        0
    )
    input_tks = torch.cat([sot_token, target_tensor], dim=-1)

    # Define the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    for step in range(5):
        # Forward pass
        predictions = model(tokens=input_tks, mel=mel)
        remove_sot = input_tks[:, 1:]  # remove sot token
        predictions = predictions[
            :, :-1, :
        ]  # remove last prediction again for alignment

        loss = criterion(predictions.transpose(1, 2), remove_sot)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    main(device)
