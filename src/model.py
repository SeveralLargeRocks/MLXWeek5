import torch
import torch.nn as nn
import whisper
from pyannote.audio import Model
from src.utils import audio_path_to_mel
from dotenv import load_dotenv
import os

class TwoTowerModel(nn.Module):
    def __init__(self, hf_token: str):
        super(TwoTowerModel, self).__init__()
        whisper_model = whisper.load_model("large")
        self.encoder = whisper_model.encoder  # Use only the encoder

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Use pyannote's speaker embedding model instead of full pipeline
        self.speaker_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=hf_token
        )
        # Freeze speaker model
        for param in self.speaker_model.parameters():
            param.requires_grad = False

        # Get dimensions
        encoder_dim = whisper_model.dims.n_audio_state
        speaker_dim = 512  # pyannote embedding dimension

        # Projection layer to combine both embeddings
        self.combine = nn.Sequential(
            nn.Linear(encoder_dim + speaker_dim, encoder_dim), nn.ReLU()
        )

        self.decoder = nn.Linear(encoder_dim, whisper_model.dims.n_vocab)

    def forward(self, mel):
        # Ensure mel spectrogram has the right shape for Whisper
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # Add batch dimension if not present
        
        # Whisper expects [batch_size, n_mels, n_frames]
        if mel.size(1) != self.encoder.conv1.in_channels:
            raise ValueError(f"Expected {self.encoder.conv1.in_channels} mel channels but got {mel.size(1)}")
            
        # Get whisper encoder features
        encoder_output = self.encoder(mel)  # Shape: [batch, seq_len, encoder_dim]
        return None

        # Get speaker embeddings
        speaker_embedding = self.speaker_model(mel)  # Shape: [batch, speaker_dim]

        # Expand speaker embedding to match sequence length
        speaker_embedding = speaker_embedding.unsqueeze(1).expand(
            -1, encoder_output.size(1), -1
        )

        # Concatenate features
        combined = torch.cat([encoder_output, speaker_embedding], dim=-1)

        # Project back to encoder dimension
        combined = self.combine(combined)

        # Pass through decoder
        logits = self.decoder(combined)  # Shape: [batch, seq_len, vocab_size]

        return logits


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    model = TwoTowerModel(hf_token=hf_token)
    mel = audio_path_to_mel("hello.wav")
    output = model(mel)
    print('done')
    # print(output.shape)
