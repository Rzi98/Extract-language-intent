from dotenv import load_dotenv
from openai import OpenAI
from typing import List
load_dotenv()

class APIWhisper:

    @classmethod
    def transcribe(cls, filename: str):
        client = OpenAI()
        audio_file = open(file=filename, mode="rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="en"
            )
        return transcript

if __name__ == '__main__':
    print(APIWhisper.transcribe(filename="../confidential_data/Arun/recording_02.wav"))
    # pass
