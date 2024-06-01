import pyaudio
import yaml
import wave
import threading
import os 
import time
from pathlib import Path
from typing import Optional
from gpt.whisper.fyp.datatypes import AudioParams

class RecordAudio:

    @staticmethod
    def read_config(filepath: Path) -> dict[AudioParams]:
        with open(file=filepath, mode='r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        
        return {
            'format': eval(config['FORMAT']),
            'channels': config['CHANNELS'], 
            'rate': config['RATE'], 
            'chunk': config['CHUNK'], 
            'output_filename': config['OUTPUT_FILENAME']
    }

    @staticmethod
    def check_uniquesave(default_savefile: Path, name: Optional[str] = None) -> Path:
        if name is None:
            basefile = default_savefile 
        else:
            placeholder = "<placeholder>"
            directory, filename = os.path.split(default_savefile)
            basefile = os.path.join(directory, placeholder, filename)
            basefile = basefile.replace('<placeholder>', name)
            basedir = '/'.join([x for x in basefile.rsplit('/')[:-1]])

            if not os.path.exists(basedir):
                os.makedirs(basedir)
                print(f"CREATED FOLDER: {basedir}")

        save_folder = '/'.join([x for x in basefile.rsplit('/')[:-1]])

        highest_batch_no = 0
        for file in os.listdir(save_folder):
            if file.endswith('.wav'):
                print(file)
                path_segment_no = len(file.split('/'))
                batch_no = int(file.rsplit('/')[path_segment_no-1].split('.')[0].split('_')[-1])
                if batch_no > highest_batch_no:
                    highest_batch_no = batch_no
                    print(highest_batch_no)
            else:
                continue

        if highest_batch_no < 10:
            return save_folder + f"/recording_0{highest_batch_no+1}.wav"
        else:
            return save_folder + f"/recording_{highest_batch_no+1}.wav"

    @classmethod
    def input_thread(cls):
        print("Press 'q' to stop recording.")
        while True:
            user_input = input()
            if user_input.lower() == 'q':
                cls.stop_recording.set()
                break

    @classmethod
    def record(cls, audio_args: dict = None, name: Optional[str] = None) -> time:
        cls.stop_recording = threading.Event()
        thread = threading.Thread(target=cls.input_thread)
        thread.daemon = True
        thread.start()

        p = pyaudio.PyAudio()

        stream = p.open(format=audio_args.format,
                        channels=audio_args.channels,
                        rate=audio_args.rate,
                        input=True,
                        frames_per_buffer=audio_args.chunk)
    
        print("Recording... (Press 'q' to stop)")
        start_t = time.perf_counter()
        cls.frames = []

        try:
            while not cls.stop_recording.is_set():
                data = stream.read(audio_args.chunk, exception_on_overflow=False)
                cls.frames.append(data)
        except KeyboardInterrupt:
            pass
    
        print("Finished recording.")
    
        stream.stop_stream()
        stream.close()
        p.terminate()

        end_t = time.perf_counter()
        duration = end_t - start_t

        audio_args.output_filename = cls.check_uniquesave(
                            default_savefile=audio_args.output_filename, 
                            name=name
                            )
        print(audio_args.output_filename)

        os.system('clear')
        esc = int(input(f"Save to {audio_args.output_filename}\
                    \nPress '1' to save, 'any' to discard: "))
        if esc != 1:
            print("Recording discarded...")
            exit()
        cls.save(p=p, audio_args=audio_args)
        return duration, audio_args.output_filename
    
    @classmethod
    def save(cls, p:pyaudio.PyAudio, audio_args: AudioParams) -> None:
        wf = wave.open(audio_args.output_filename, 'wb')
        wf.setnchannels(audio_args.channels)
        wf.setsampwidth(p.get_sample_size(audio_args.format))
        wf.setframerate(audio_args.rate)
        wf.writeframes(b''.join(cls.frames))
        wf.close()