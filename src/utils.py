import torch
import whisper
from pyannote.audio import Pipeline
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_pyannote_diarization_model(hf_token: str) -> Pipeline:
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )


def get_diarization(model, audio_path: str) -> list[tuple[float, float, str]]:
    diarization = model(audio_path)
    return diarization


def audio_path_to_mel(audio_path: str, device: str = "cpu") -> torch.Tensor:
    waveform = whisper.load_audio(audio_path)
    waveform_padded = whisper.pad_or_trim(waveform)
    mel = whisper.log_mel_spectrogram(waveform_padded).to(device).unsqueeze(0)
    return mel


def text_to_input_tks(
    text: str, tokenizer: whisper.tokenizer.Tokenizer, device: str = "cpu"
) -> torch.Tensor:
    target_ids = tokenizer.encode(text)
    sot_token = torch.tensor(
        [tokenizer.sot], dtype=torch.long, device=device
    ).unsqueeze(0)
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(
        0
    )
    input_tks = torch.cat([sot_token, target_tensor], dim=-1)

    return input_tks

def text_batch_to_input_tks(
    text: str, tokenizer: whisper.tokenizer.Tokenizer, device: str = "cpu"
) -> torch.Tensor:
    target_ids = tokenizer.encoding.encode_batch(text)
    input_tks = [[tokenizer.sot] + target_ids for target_ids in target_ids]

    return input_tks


def get_loss(
    predictions: torch.Tensor,
    input_tks: torch.Tensor,
    criterion: torch.nn.CrossEntropyLoss,
) -> torch.Tensor:
    remove_sot = input_tks[:, 1:]  # remove sot token
    predictions = predictions[:, :-1, :]  # remove last prediction again for alignment

    loss = criterion(predictions.transpose(1, 2), remove_sot)
    return loss


def transcribe(model, audio_path: str, word_timestamps: bool = False) -> dict:
    result = model.transcribe(audio_path, word_timestamps=word_timestamps)
    return result

def get_training_kit() -> tuple[
    str,
    whisper.model.Whisper,
    whisper.tokenizer.Tokenizer,
    torch.optim.Optimizer,
    torch.nn.CrossEntropyLoss,
]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("tiny.en", device=device)
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    return device, model, tokenizer, optimizer, criterion
