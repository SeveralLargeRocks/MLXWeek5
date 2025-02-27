from src.utils import transcribe
import whisper

model = whisper.load_model("small")  # load smaller model for speed
result = transcribe(model, "modern_wisdom_alain_de_botton.wav")
print("result: ", result)

print('my bro')
