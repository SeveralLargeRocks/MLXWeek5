import torch
import torch.nn as nn
import whisper
from pyannote.audio import Model, Pipeline
from src.utils import audio_path_to_mel
from dotenv import load_dotenv
import os

class TwoTowerModel(nn.Module):
    def __init__(self, hf_token: str):
        super(TwoTowerModel, self).__init__()
        whisper_model = whisper.load_model("small.en")
        self.encoder = whisper_model.encoder  # Use only the encoder

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Use pyannote's speaker embedding model instead of full pipeline
        # self.speaker_model = Model.from_pretrained(
        #     "pyannote/embedding", use_auth_token=hf_token
        # )
        self.speaker_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=hf_token
        )

        # Freeze speaker model
        # for param in self.speaker_model.parameters():
        #     param.requires_grad = False

        # Get dimensions
        encoder_dim = whisper_model.dims.n_audio_state
        self.speaker_dim = 256  # pyannote embedding dimension

        # Projection layer to combine both embeddings
        self.combine = nn.Sequential(
            nn.Linear(encoder_dim + self.speaker_dim, encoder_dim), nn.ReLU()
        )

        self.decoder = nn.Linear(encoder_dim, whisper_model.dims.n_vocab)

    def forward(self, waveform):
        waveform_padded = whisper.pad_or_trim(waveform)

        device = next(model.parameters()).device
        mel = whisper.log_mel_spectrogram(waveform_padded).to(device).unsqueeze(0)
        # # Ensure mel spectrogram has the right shape for Whisper
        # if mel.dim() == 2:
        #     mel = mel.unsqueeze(0)  # Add batch dimension if not present
        
        # # Whisper expects [batch_size, n_mels, n_frames]
        # if mel.size(1) != self.encoder.conv1.in_channels:
        #     raise ValueError(f"Expected {self.encoder.conv1.in_channels} mel channels but got {mel.size(1)}")
            
        # Get whisper encoder features
        encoder_output = self.encoder(mel)  # Shape: [batch, seq_len, encoder_dim]
        print(encoder_output.shape)

        audio_token_length = 20 # 20ms

        waveform_tensor = torch.tensor(waveform).unsqueeze(0)
        # Get speaker embeddings
        # speaker_embedding = self.speaker_model(waveform_tensor)  # Shape: [batch, speaker_dim]
        diarization, embeddings = self.speaker_pipeline({ 'waveform': waveform_tensor, 'sample_rate': 16000 }, return_embeddings=True)
        print(diarization, embeddings)

        # for batch in encoder_output:
        #     for audio_token in batch:
        #         matching_segment = # find matching semgnet
        #         audio_token = torch.concat(audio_token, matching_segment) 

        addition_matrix = torch.tensor([], device=device)
        
        index_lookup = {
            'SPEAKER_00': 0,
            'SPEAKER_01': 1
        }

        prev_finish_time = 0
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            print('i')
            time_since_last_segment_end = segment.start - prev_finish_time

            empty_rows = (time_since_last_segment_end * 1000) / audio_token_length
            actual_empty_rows = torch.zeros(int(empty_rows), self.speaker_dim)
            print('empty rows shape:', actual_empty_rows.shape)

            index = index_lookup[speaker]
            segment_speaker_embedding = torch.tensor(embeddings[index])
            segment_duration_ms = (segment.end - segment.start) * 1000
            num_tracks_in_segment = segment_duration_ms / audio_token_length
            # print(num_tracks_in_segment)
            # print(segment_speaker_embedding.shape)
            repeated = segment_speaker_embedding.repeat(int(num_tracks_in_segment), 1)
            print('repeated shape:', repeated.shape)

            addition_matrix = torch.cat((addition_matrix, actual_empty_rows), 0)
            addition_matrix = torch.cat((addition_matrix, repeated), 0)
            prev_finish_time = segment.end
            print(f"{segment.start} - {segment.end}: {speaker}")

        remaing_empty_rows_to_append = 1500 - addition_matrix.shape[0]
        final_empty_rows = torch.zeros(int(remaing_empty_rows_to_append), self.speaker_dim)
        addition_matrix = torch.cat((addition_matrix, final_empty_rows), 0)
        print('addition matrix shape:', addition_matrix.shape)

        return None

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
    waveform = whisper.load_audio("extract.wav")
    output = model(waveform)
    print('done')
    # print(output.shape)
