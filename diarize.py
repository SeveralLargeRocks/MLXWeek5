import huggingface_hub
import pyannote.audio
import pyannote.database
import pyannote.core
import os
import torchaudio
from pyannote.core import Segment
from io import BytesIO
import subprocess

SAMPLE_RATE = 16_000

dirname = os.path.dirname(__file__)

huggingface_hub.login(token=os.environ.get('HUGGINGFACE_TOKEN', None))

# { start_ms: int, text: str, speaker: str }
transcript = []

def predict_speakers(waveform, start_time, max_speakers):
    print('Loading pipeline...')

    pipeline = pyannote.audio.Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)

    print('> done')

    # segmenter = pyannote.audio.Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)

    audio_in_memory = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

    # inference = pyannote.audio.Inference(segmenter, step=1)

    print('Running pipeline...')

    diarization, embeddings = pipeline(audio_in_memory, return_embeddings=True, min_speakers=1, max_speakers=max_speakers)

    print('embeddings:')
    print(embeddings)

    for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
        print(speech_turn, track, speaker)

def convert_webm_to_wav(webm_data: bytes) -> bytes:
    # Create a temporary file for the WebM input
    with open("temp.webm", "wb") as f:
        f.write(webm_data)
    
    # Convert WebM to WAV using ffmpeg
    output_file = "temp.wav"
    try:
        subprocess.run([
            "ffmpeg", "-i", "temp.webm", 
            "-acodec", "pcm_s16le",  # Convert to 16-bit PCM WAV
            output_file
        ], check=True)
    except subprocess.CalledProcessError as e:
        print('no working', e)
    
    # Read the converted WAV file
    with open(output_file, "rb") as f:
        wav_data = f.read()
    
    # Clean up temporary files
    os.remove("temp.webm")
    os.remove(output_file)
    
    return wav_data

def handle_socket_message(bytes):
    # data = BytesIO(bytes)

    wav = convert_webm_to_wav(bytes)

    waveform, sample_rate = torchaudio.load(wav, format="webm")
    
    print('predicting:', waveform)
    # predict_speakers(waveform, 0, 4)
    print('done predicting')

if __name__ == "__main__":

    AUDIO_FILE = os.path.join(dirname, "extract2.mp3")

    waveform, sample_rate = torchaudio.load(AUDIO_FILE)

    predict_speakers(waveform, 0, 4)