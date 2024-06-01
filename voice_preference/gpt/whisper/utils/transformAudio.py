from pydub import AudioSegment
from pathlib import Path
import os
import re

candidate: str = "Heather"
directory: Path = Path(f"../audio_files/{candidate}")

def extract_digits_between_letter_and_dot(input_string):
    match = re.search(r'[A-Za-z](\d+)\.', input_string)
    # print(f"MATCH: {match}")
    if match:
        return match.group(1)
    else:
        return None

def removeZoneFiles(directory: Path):
    for file in os.listdir(directory):
        if not file.endswith(".m4a") and not file.endswith(".wav"):
            os.remove(os.path.join(directory, file))
            print(f"Removed {file}")
        if file.endswith(".m4a"):
            idx = extract_digits_between_letter_and_dot(file)
            if int(idx) < 10:
                idx = f"0{idx}"
            filepath = os.path.join(directory, file)
            m4a_audio = AudioSegment.from_file(filepath, format="m4a")
            m4a_audio.export(os.path.join(directory, f'recording_{idx}.wav'), format="wav")
            os.remove(filepath)
            print(f"Converted {file} to .wav")

def renamefile(directory: Path):
    for file in sorted(os.listdir(directory)):
        idx = int(file.split('_')[1].split('.')[0])
        idx -= 1
        new_filename = f"recording_{str(idx).zfill(2)}.wav"
        print(f"{file} -> {new_filename}")
        os.rename(os.path.join(directory, file), os.path.join(directory, new_filename))
        print("\nRenamed COMPLETE!")

if __name__ == '__main__':
    renamefile(directory='../audio_files/JN')
    pass
    # removeZoneFiles(directory=directory)