import re
import os
import pandas
import os
import whisper

dirname = os.path.dirname(__file__)

def parse_diarized_file(filename):
    """Parse the diarized file into a list of (start_time, end_time, speaker, text) tuples."""
    diarized_segments = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # Extract time range, speaker, and text
                match = re.match(r'\[(\d+\.\d+)s\s*->\s*(\d+\.\d+)s\]\s*(SPEAKER_\d+):\s*(.*)', line)
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    speaker = match.group(3)
                    text = match.group(4)
                    diarized_segments.append((start_time, end_time, speaker, text))
    return diarized_segments

def generate_output_string(diarized_segments):
    """Generate the output string with <|startoflm|> for speaker changes and <split> every 30 seconds based on absolute timestamps."""
    output_text = ""
    previous_speaker = None
    next_split_time = 30.0  # The next absolute timestamp where <split> should be inserted

    tokenizer = whisper.tokenizer.get_tokenizer(True)
    
    for start_time, end_time, speaker, text in diarized_segments:
        # Insert <|startoflm|> if the speaker changes
        if speaker != previous_speaker:
            if previous_speaker is not None:  # Skip <|startoflm|> for the first speaker
                output_text += tokenizer.sot_lm
        
        words = text.split()
        segment_duration = end_time - start_time
        words_per_second = len(words) / segment_duration if segment_duration > 0 else 0

        # Process each word while keeping track of its estimated absolute time
        word_timestamps = []
        for i, word in enumerate(words):
            word_time = start_time + (i / words_per_second) if words_per_second > 0 else start_time
            word_timestamps.append((word_time, word))
        
        for word_time, word in word_timestamps:
            # Insert <split> exactly at absolute 30-second intervals
            while word_time >= next_split_time:
                output_text += " <split>"
                next_split_time += 30  # Move to the next absolute 30-second mark
            
            output_text += f" {word}"
        
        # Update previous speaker for the next iteration
        previous_speaker = speaker
    
    return output_text

def split_text_by_token(text):
    """Splits the text by <split> and assigns an ID to each segment."""
    segments = text.split(" <split>")
    split_dict = {i + 1: segment.strip() for i, segment in enumerate(segments) if segment.strip()}
    return split_dict

def process_diarized_file(filename):
    """Main function to process the diarized file and output the long string."""
    diarized_segments = parse_diarized_file(filename)
    output_string = generate_output_string(diarized_segments)
    return output_string
    
def associate_audio_with_text(audio_folder, audio_file, text_dict):
    """Associates each audio file in the folder with its corresponding text ID."""
    audio_mapping = {}

    # Regex to extract segment number from filenames (matching specific audio_file prefix)
    pattern = re.compile(re.escape(audio_file) + r'_segment_(\d+)\.wav')

    for filename in os.listdir(audio_folder):
        match = pattern.search(filename)
        if match:
            segment_id = int(match.group(1))  # Extract segment number
            if segment_id in text_dict:
                audio_mapping[filename] = text_dict[segment_id]

    return audio_mapping

if __name__ == "__main__":
    #generate train output
    diarized_file = os.path.join(dirname, "modern_wisdom_alain_de_botton_combined.txt")  # Replace with your actual file path
    output_string = process_diarized_file(diarized_file)
    
    splitted=split_text_by_token(output_string)
        
    audio_folder = os.path.join(dirname, "split")

    result = associate_audio_with_text(audio_folder, "modern_wisdom_alain_de_botton", splitted)

    df = pandas.DataFrame.from_dict(result, orient='index')
    df.to_csv('output_train.csv', header=False)
    print("output_train.csv generated")

    #generate validation output
    diarized_file = os.path.join(dirname, "modern_wisdom_dr_julie_smith_combined.txt")  # Replace with your actual file path
    output_string = process_diarized_file(diarized_file)
    
    splitted=split_text_by_token(output_string)
        
    audio_folder = os.path.join(dirname, "split")

    result = associate_audio_with_text(audio_folder, "modern_wisdom_dr_julie_smith", splitted)

    df = pandas.DataFrame.from_dict(result, orient='index')
    df.to_csv('output_validation.csv', header=False)
    print("output_validation.csv generated")









