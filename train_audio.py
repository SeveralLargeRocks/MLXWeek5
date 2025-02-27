
import soundfile as sf

# Load the audio file
audio, sample_rate = sf.read("split/modern_wisdom_alain_de_botton.wav")

# Split the audio into segments
segment_length = 30  # 30 seconds per segment
num_segments = int(len(audio) / (segment_length * sample_rate))

for i in range(num_segments):
    start_time = i * segment_length * sample_rate
    end_time = (i + 1) * segment_length * sample_rate
    segment = audio[start_time:end_time]

    # Save the segment to a file
    filename = f"split/modern_wisdom_alain_de_botton_segment_{i+1}.wav"
    sf.write(filename, segment, sample_rate)