import shutil
from pathlib import Path

def zip_directory(inputpath: Path, output_zip_path: Path):
    shutil.make_archive(output_zip_path, 'zip', inputpath)

if __name__ == "__main__":

    zip_directory(inputpath=Path('../audio_files'), output_zip_path=Path('../audio_files'))
