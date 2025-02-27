import os
from dotenv import load_dotenv
from src.utils import get_pyannote_diarization_model

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


diarization_pipeline = get_pyannote_diarization_model(hf_token)

diarization = diarization_pipeline("modern_wisdom_alain_de_botton.wav")
for segment, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{segment.start:.2f} - {segment.end:.2f}: {speaker}")

print('my bro')
