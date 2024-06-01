from transformers import pipeline
import torch

class TextToSpeech:

    @classmethod
    def load_model(cls):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cls.pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
        device=device,
        )

    @classmethod
    def transcribe(cls, audiofile):
        return cls.pipe(audiofile, batch_size=8)["text"]
    
if __name__ == '__main__':
    import time, os

    print("LOADING MODEL...")
    TextToSpeech.load_model()
    print("MODEL LOADED")

    time.sleep(2)
    os.system('clear')

    while input("Do you want to continue? [Y/N] ").lower().startswith('y'):
        dataset = input("Enter the dataset: ")
        filenum = input("Enter the file number: ")
        print(f"../audio_files/{dataset}/recording_{filenum}.wav")
        transcribed_txt = TextToSpeech.transcribe(audiofile=f"../audio_files/{dataset}/recording_{filenum}.wav")
        print(f"TRANSCRIPTION: {transcribed_txt}")

    print()
    print("EXITING...") 