import re

def parse_diarized_file(filename):
    """Parse the diarized file into a list of (start_time, end_time, speaker) tuples."""
    diarized_segments = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # Extract time range and speaker
                match = re.match(r'(\d+\.\d+)\s*-\s*(\d+\.\d+):\s*(SPEAKER_\d+)', line)
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    speaker = match.group(3)
                    diarized_segments.append((start_time, end_time, speaker))
    return diarized_segments

def parse_timestamped_file(filename):
    """Parse the timestamped file into a list of (start_time, end_time, text) tuples."""
    timestamped_segments = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # Extract time range and text
                match = re.match(r'\[(\d+\.\d+)s\s*->\s*(\d+\.\d+)s\]\s*(.*)', line)
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    text = match.group(3)
                    timestamped_segments.append((start_time, end_time, text))
    return timestamped_segments

def find_speaker_for_segment(segment_start, segment_end, diarized_segments):
    """Find the speaker for a given time segment."""
    # Find the speaker with the most overlap with this segment
    max_overlap = 0
    best_speaker = "UNKNOWN"
    
    for d_start, d_end, speaker in diarized_segments:
        # Calculate overlap
        overlap_start = max(segment_start, d_start)
        overlap_end = min(segment_end, d_end)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = speaker
    
    return best_speaker

def combine_files(diarized_file, timestamped_file, output_file):
    """Combine the diarized and timestamped files."""
    diarized_segments = parse_diarized_file(diarized_file)
    timestamped_segments = parse_timestamped_file(timestamped_file)
    
    with open(output_file, 'w') as out_file:
        for start_time, end_time, text in timestamped_segments:
            speaker = find_speaker_for_segment(start_time, end_time, diarized_segments)
            # Format: [START -> END] SPEAKER: Text
            out_file.write(f"[{start_time:.2f}s -> {end_time:.2f}s] {speaker}: {text}\n")

if __name__ == "__main__":
    combine_files("modern_wisdom_diarized.txt", "modern_wisdom_timestamped.txt", "modern_wisdom_combined.txt")
    print("Files combined successfully! Output written to modern_wisdom_combined.txt")