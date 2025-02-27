from src.utils import transcribe
import whisper

model = whisper.load_model("small")
# Set word_timestamps=True to get timestamp data
result = transcribe(model, "modern_wisdom_alain_de_botton.wav", word_timestamps=True)

# Print segments with timestamps
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.2f}s -> {end:.2f}s] {text}")

print('my bro')
