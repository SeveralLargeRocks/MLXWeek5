import torch
import torch.nn as nn
import whisper
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os
import numpy as np
import gc


class TwoTowerModel(nn.Module):
    def __init__(self, hf_token: str):
        super(TwoTowerModel, self).__init__()
        whisper_model = whisper.load_model("small.en")

        self.is_multilingual = whisper_model.is_multilingual
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

        self.tokenizer = whisper.tokenizer.get_tokenizer(self.is_multilingual)

        self.decoder = whisper_model.decoder

    def forward(self, waveform, token_ids):
        waveform_padded = whisper.pad_or_trim(waveform)
        print('waveform_padded.shape: ', waveform_padded.shape)

        device = next(self.parameters()).device
        mel = whisper.log_mel_spectrogram(waveform_padded).to(device)

        print('mel shape', mel.shape)
            
        # Get whisper encoder features
        encoder_output = self.encoder(mel)  # Shape: [batch, seq_len, encoder_dim]
    
        print(encoder_output)

        audio_token_length = 20 # 20ms

        waveform_tensor = torch.tensor(waveform, device=device)
        print('waveform_tenso.shaper: ', waveform_tensor.shape)

        # Get diarization & speaker embeddings
        diarization, embeddings = self.speaker_pipeline({ 'waveform': waveform_tensor, 'sample_rate': 16000 }, return_embeddings=True)

        # matrix to be concatted with the whisper encoder output
        who_matrix = torch.tensor([], device=device)
        
        # TODO: do some string manipulation stuff to avoid need for lookup
        index_lookup = {
            'SPEAKER_00': 0,
            'SPEAKER_01': 1,
            'SPEAKER_02': 2,
            'SPEAKER_03': 3,
            'SPEAKER_04': 4
        }

        prev_segment_end = 0
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_time = segment.start
            if start_time < prev_segment_end:
                # if someone interrupts, preserve the initial speaker until they stop speaking
                start_time = prev_segment_end

            if segment.end < prev_segment_end:
                # if entire segment is speaking over previous speaker, ignore them completely
                continue

            time_since_last_segment_end = start_time - prev_segment_end

            if time_since_last_segment_end > 0:
                # fill the rows between each detected speech segment (silence) with zeros
                empty_rows = (time_since_last_segment_end * 1000) / audio_token_length
                actual_empty_rows = torch.zeros(int(empty_rows), self.speaker_dim).to(device)

                who_matrix = torch.cat((who_matrix, actual_empty_rows), 0)

            # then fill the rows for the speech segment with the embedding for that speaker (repeated)
            index = index_lookup[speaker]
            segment_speaker_embedding = torch.tensor(embeddings[index], device=device)

            segment_duration_ms = (segment.end - start_time) * 1000
            num_tracks_in_segment = segment_duration_ms / audio_token_length
            repeated = segment_speaker_embedding.repeat(int(num_tracks_in_segment), 1)

            who_matrix = torch.cat((who_matrix, repeated), 0)

            prev_segment_end = segment.end

        # fill the remaining rows with zeros (corresponds to whisper's own padding to 30 seconds)
        remaing_empty_rows_to_append = 1500 - who_matrix.shape[0]
        final_empty_rows = torch.zeros(int(remaing_empty_rows_to_append), self.speaker_dim).to(device)
        who_matrix = torch.cat((who_matrix, final_empty_rows), 0)

        # TODO: who_matrix should be batched
        who_matrix = who_matrix.unsqueeze(0)

        who_what_matrix = torch.cat((encoder_output, who_matrix), -1)

        # Project back to encoder dimension
        combined_audio = self.combine(who_what_matrix)

        logits = self.decoder(token_ids, combined_audio)  # Shape: [batch, seq_len, vocab_size]

        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    model = TwoTowerModel(hf_token=hf_token).to(device)
    model.eval()  # Set to evaluation mode
    
    waveform = whisper.load_audio("extract.wav")
    waveform_batch = np.expand_dims(waveform, axis=0)
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
    
    # Start with just the necessary tokens
    prompt = [tokenizer.sot]  # Start of transcript token
    if model.is_multilingual:
        prompt.append(tokenizer.language_token("en"))
    prompt.append(tokenizer.no_timestamps)  # Add no timestamps token
    
    tokens = torch.tensor([prompt]).to(device)
    
    with torch.no_grad():
        # Generate tokens until we hit max length or end token
        max_len = 448  # Whisper's max length
        while tokens.shape[-1] < max_len:
            print('tokens shape', tokens.shape)
            logits = model(waveform_batch, tokens)
            next_token = torch.argmax(logits[0, -1])

            if next_token == tokenizer.eot:  # Break if we hit end of transcript
                break
                
            tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
            print("Generated token:", tokenizer.decode([next_token.item()]))
    
    # Decode the full sequence
    text = tokenizer.decode(tokens[0].tolist())
    print("\nFinal transcript:", text)
