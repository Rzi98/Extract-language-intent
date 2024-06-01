import os
import time
import logging
import asyncio
import aiohttp 
import json
import re
from pathlib import Path
from typing import List, Optional

from gpt.whisper.fyp.datatypes import AudioParams
from gpt.whisper.fyp.record import RecordAudio
# from gpt.whisper.fyp.models import TextToSpeech
from gpt.whisper.fyp.apiModel import APIWhisper
from gpt.module.endpoints import GPT, generate_class_list, generate_intensity_list
from gpt.whisper.fyp.whisperCorrection import WhisperCorrection


CONFIG_PATH: Path = Path("./gpt/whisper/config/params.yaml")
LOGS_PATH: Path = Path("./gpt/whisper/logs/main.log")

# cache = {}
def set_logger(logger_filepath: Path):

    if not os.path.exists(logger_filepath):
        os.makedirs(logger_filepath)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    file_handler = logging.FileHandler(logger_filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def voice_record(candidate_name: str) -> tuple[float, str]:
    audio_args = AudioParams(**RecordAudio.read_config(CONFIG_PATH))
    duration, last_save = RecordAudio.record(audio_args=audio_args, name=candidate_name)
    print(f"Recording saved to {last_save}")
    print(f"Recording duration: {duration} seconds")
    return last_save

def save_inference(filepath: Path, inference: dict):
    if os.path.exists(filepath):
        with open(file=filepath, mode='r') as json_file:
            data: list = json.load(json_file)
            idx = len(data) + 1
            inference['id'] = idx
            data.append(inference)
    else:
        os.makedirs(filepath.parent, exist_ok=True)
        inference['id'] = 1
        data = [inference]
    with open(file=filepath, mode='w') as json_file:
        json.dump(data, json_file, indent=4)


async def main(raw_transcribed:Optional[str]=None, rectified_transcription: Optional[str]=None, candidate: Optional[str]='Samples', prompt_type: Optional[str]=None):  
    start = time.perf_counter()
    gpt: GPT = GPT(model="gpt-4-1106-preview")
    total_tokens: int = 0 

    async with aiohttp.ClientSession() as session:
        tasks = [
                    asyncio.create_task(gpt.get_obj_list(session=session, user_msg=rectified_transcription)),
                    asyncio.create_task(gpt.split_commands(session=session, user_msg=rectified_transcription))
                ]
        task1, task2 = await asyncio.gather(*tasks)
        object_list: List[str] = task1[0]
        split_cmd: List[str] = task2[0]["split"] 
        trajectory_type: List[str] = task2[0]["type"] 
        total_tokens += (task1[-1] + task2[-1])

        dynamic_features = await gpt.get_dynamic_features(obj_list=object_list, trajectory_list=trajectory_type)

    async with aiohttp.ClientSession() as session:
        
        tasks = [
                    asyncio.create_task(gpt.get_inference(session=session, split_cmd=split_cmd, dynamic_list=dynamic_features)),
                    asyncio.create_task(gpt.get_intensity(session=session, split_cmd=split_cmd))
                ]
        
        task3, task4 = await asyncio.gather(*tasks)
        classification: List[str] = task3[0]["output"]
        total_tokens += (task3[-1] + task4[-1])

        subtasks = [
                    asyncio.create_task(generate_class_list(task=task3)),
                    asyncio.create_task(generate_intensity_list(split_task=task2, intensity_task=task4, type_task=trajectory_type))
                ]
        
        class_list: List[str]
        intensity_list: List[str]

        class_list, intensity_list = await asyncio.gather(*subtasks)
        
    end: float = time.perf_counter()
    
    print(f"RAW TRANSCRIPTION: '{raw_transcribed}'")
    print(f"RECTIFIED INPUT: '{rectified_transcription}'")
    print(f"OBJECT LIST: {object_list}")
    print(f"DYNAMIC FEATURE: {dynamic_features}")
    print(f"SPLIT COMMAND: {split_cmd}")
    print(f"TRAJECTORY TYPE: {trajectory_type}")
    print(f"CLASSIFICATION: {classification}")
    print(f"CLASS LIST: {class_list}")
    print(f"INTENSITY LIST: {intensity_list}")
    print(f"TOKEN: {total_tokens}")
    print(f"CHAT MODEL INFERENCE TIME: {end - start:0.2f} seconds")

    inference = {
        "raw_input": raw_transcribed,
        "rectified_input": rectified_transcription,
        "object_list": object_list,
        "dynamic_feature": dynamic_features,
        "split_command": split_cmd,
        "trajectory_type": trajectory_type,
        "classification": classification,
        "class_list": class_list,
        "intensity_list": intensity_list,
        "token": total_tokens,
        "inference_time": end - start
    }
    
    if prompt_type == "semantic":
        save_inference(filepath=Path(f"./gpt/whisper/results/semantic/{candidate}.json"), inference=inference)
    elif prompt_type == "sound":
        save_inference(filepath=Path(f"./gpt/whisper/results/sound/{candidate}.json"), inference=inference)
    else:
        print(f"TEST RESULT: {inference}")
        raise ValueError("Invalid prompt type. Please specify either 'semantic' or 'sound'.")


def recordAndTranscribe(candidate:Optional[str]="Raizee", gt_vision_list:Optional[list]=[], prompt_type:Optional[str]=None):
    while True:
        os.system('clear')
        last_audiofile = voice_record(candidate_name=candidate)

        # start_t = time.perf_counter()

        raw_text = APIWhisper.transcribe(filename=last_audiofile)

        if gt_vision_list:
            text = WhisperCorrection.rectify_transcription(transcription=raw_text, vision_list=gt_vision_list, prompt_type=prompt_type)

        # end_t = time.perf_counter()
        asyncio.run(main(raw_transcribed=raw_text, rectified_transcription=text, candidate=candidate, prompt_type=prompt_type))

        if input("Do you want to continue? [Y/N] ").lower().startswith('n'):
            print("Exiting...")
            break

def transcribeOnly(candidate: str, recording_num:Optional[str]=None, gt_vision_list:Optional[list]=[], prompt_type:Optional[str]=None):
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else None


    directory = f"./gpt/whisper/confidential_data/{candidate}"

    sorted_files = sorted(
        [file for file in os.listdir(directory) if file.endswith(".wav") or file.endswith(".aac")],
        key=extract_number
    )
    for idx, file in enumerate(sorted_files):
        last_audiofile = os.path.join(directory, file)

        start_t = time.perf_counter()
        if idx + 1 >= int(recording_num):
            print(f"Transcribing file #{idx + 1}: {last_audiofile}...")

            raw_text = APIWhisper.transcribe(filename=last_audiofile)
            print(f"RAW TRANSCRIPTION: {raw_text}")

            if gt_vision_list:
                text = WhisperCorrection.rectify_transcription(transcription=raw_text, vision_list=gt_vision_list, prompt_type=prompt_type)
                print(f"RECTIFIED TRANSCRIPTION: {text}")

            end_t = time.perf_counter()
            print(f"TRANSCRIPTION: {text} | Time taken: {end_t - start_t:0.2f} seconds")
            asyncio.run(main(raw_transcribed=raw_text, rectified_transcription=text, candidate=candidate, prompt_type=prompt_type))
            print("="*150)
    
    print("Transcription complete & result recorded.")


def test_from_text(text:str):
        asyncio.run(main(raw_transcribed=text, rectified_transcription=text, candidate="Sample", prompt_type=None))     
    

if __name__ == '__main__':
    ### FOR RECORDING ###
    # recordAndTranscribe(candidate="Sample", gt_vision_list=["scissors", "mug", "vase", "bottle", "bowl", "orange", "laptop", "banana", "flower"], prompt_type="semantic")

    ### FOR TRANSCRIBING RECORDED FILES ###
    transcribeOnly(candidate="Heather", recording_num='1', gt_vision_list=["scissors", "mug", "vase", "bottle", "bowl", "orange", "laptop", "banana", "flower"], prompt_type="sound")

    ## RUN EVAL ##
    # for name in os.listdir("./gpt/whisper/confidential_data"):
    #     # if name in ["XinZhi", "MinThet", "Malthus", "JN", "Meiwen", "RuiQian", "ChorTeng", "RuiBin", "JunWei", "MinnSet"]:
    #     #     continue
    #     try:
    #         transcribeOnly(candidate=name, recording_num='1', gt_vision_list=["scissors", "mug", "vase", "bottle", "bowl", "orange", "laptop", "banana", "flower"], prompt_type="semantic")
    #     except Exception as e:
    #         print("\n")
    #         print("="*150)
    #         print(f"Error: {e} for {name}")
    #         print("="*150)
    #         print("\n")
    #         exit()


    ### FOR TESTING W/O AUDIO ###
    # test_from_text(text="Move front but keep a distance to bottles")