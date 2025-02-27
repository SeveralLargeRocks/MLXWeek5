import datasets

train_dataset = datasets.load_dataset(
        path="librispeech_asr",
        name="clean",
        split="train.100",
        cache_dir="./data",
        trust_remote_code=True,
    )

# sample = valid_dataset.select(['speaker_id', 'sentence'])

# Check for files with multiple speaker IDs
file_speakers = {}
for item in train_dataset:
    file = item['file']
    speaker_id = item['speaker_id']
    if file not in file_speakers:
        file_speakers[file] = set()
    file_speakers[file].add(speaker_id)

# Find conflicts where a single file has multiple speakers
conflicts = {file: list(speakers) for file, speakers in file_speakers.items() if len(speakers) > 1}

if conflicts:
    print("Files with multiple speakers:")
    for file, speakers in conflicts.items():
        print(f"File: {file}, Speakers: {speakers}")
else:
    print("No files with multiple speakers found")


# speaker_counts = {}
# for speaker_id in valid_dataset["speaker_id"]:
#     speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1


# print(dict(sorted(speaker_counts.items(), key=lambda item: item[1], reverse=True)))



