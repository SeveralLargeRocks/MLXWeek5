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

        # Use pipeline for now (TODO: can we do this faster using the separate models?)
        self.speaker_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=hf_token
        )

        # TODO: ensure pipeline is frozen already
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
            
        # Get whisper encoder features
        encoder_output = self.encoder(mel)  # Shape: [batch, seq_len, encoder_dim]
        print(encoder_output.shape)

        audio_token_length = 20 # 20ms

        waveform_tensor = torch.tensor(waveform).unsqueeze(0)

        # Get diarization & speaker embeddings
        diarization, embeddings = self.speaker_pipeline({ 'waveform': waveform_tensor, 'sample_rate': 16000 }, return_embeddings=True)

        # matrix to be concatted with the whisper encoder output
        who_matrix = torch.tensor([], device=device)
        
        # TODO: do some string manipulation stuff to avoid need for lookup
        index_lookup = {
            'SPEAKER_00': 0,
            'SPEAKER_01': 1
        }

        prev_segment_end = 0
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            time_since_last_segment_end = segment.start - prev_segment_end

            # fill the rows between each detected speech segment (silence) with zeros
            empty_rows = (time_since_last_segment_end * 1000) / audio_token_length
            actual_empty_rows = torch.zeros(int(empty_rows), self.speaker_dim)

            who_matrix = torch.cat((who_matrix, actual_empty_rows), 0)

            # then fill the rows for the speech segment with the embedding for that speaker (repeated)
            index = index_lookup[speaker]
            segment_speaker_embedding = torch.tensor(embeddings[index])
            segment_duration_ms = (segment.end - segment.start) * 1000
            num_tracks_in_segment = segment_duration_ms / audio_token_length
            repeated = segment_speaker_embedding.repeat(int(num_tracks_in_segment), 1)

            who_matrix = torch.cat((who_matrix, repeated), 0)

            prev_segment_end = segment.end

        # fill the remaining rows with zeros (corresponds to whisper's own padding to 30 seconds)
        remaing_empty_rows_to_append = 1500 - who_matrix.shape[0]
        final_empty_rows = torch.zeros(int(remaing_empty_rows_to_append), self.speaker_dim)
        who_matrix = torch.cat((who_matrix, final_empty_rows), 0)

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
