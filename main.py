"""Speech recognition using OpenAI's Whisper model."""

import torch
import whisper


def main(device: str = "cpu") -> None:
    """Initialize and load Whisper model on specified device."""

    model = whisper.load_model("tiny.en", device=device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    main(device)
