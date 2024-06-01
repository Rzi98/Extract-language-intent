from dotenv import load_dotenv
from functools import reduce  
import asyncio
import aiohttp 
import time
import re
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import (List, Optional, 
                    TypedDict, Union, Tuple)
from tenacity import (retry, stop_after_attempt, 
                      wait_fixed, retry_if_exception_type)

from module.prompts import *
from module.exceptions import CustomError
from module.datatypes import model_version

load_dotenv()

TOKEN = os.getenv('OPENAI_API_KEY')
URL = "https://api.openai.com/v1/chat/completions"
CONTENT_TYPE = "application/json"
HEADER: dict = {
                        "Authorization": f"Bearer {TOKEN}",
                        "Content-Type": CONTENT_TYPE
                    }

class GPT:
    def __init__(self, model: Optional[model_version]='gpt-4-1106-preview') -> None:
        self.model = model  

    @staticmethod
    def get_body(model: model_version, user_msg: Union[str, List[str]], system_prompt: str) -> dict:
        body: dict
        body = {
                "model": model,
                "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_msg
                        }      
                    ],
                    "temperature": 0,
                    "max_tokens": 512,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "response_format": { "type": "json_object" },
                    "seed": 0
                }
        return body
    
    async def get_distance_features(self, obj_list: List[str]) -> List[str]:
        dist_list: list = ['distance increase', 'distance decrease']
        distance_features: list = [f"{item1} {item2}" for item1 in obj_list for item2 in dist_list]
        return distance_features
    
    async def get_speed_features(self) -> Union[List[str], None]:
            speed_list: list = ['speed increase', 'speed decrease']
            return speed_list
        
    async def get_dynamic_features(self, obj_list: List[str], trajectory_list: List[str]) -> List[str]:
        if 'SPEED' in trajectory_list:
            tasks = [asyncio.create_task(self.get_distance_features(obj_list)), 
                     asyncio.create_task(self.get_speed_features())]
            distance_features, speed_features = await asyncio.gather(*tasks)
            return distance_features + speed_features
        return await self.get_distance_features(obj_list=obj_list)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(40),retry=retry_if_exception_type(CustomError))
    async def get_obj_list(self, session: aiohttp.ClientSession, user_msg: str) -> Tuple[List, List, int]:
        body: dict = self.get_body(model=self.model, user_msg=user_msg, system_prompt=dynamic_feature_prompt)

        async with session.post(URL, headers=HEADER, data=json.dumps(body)) as response:
    
            if response.status == 200:
                content:json = await response.json()
                obj_list: list = json.loads(content["choices"][0]["message"]["content"])['obj_list']
                token: int = content["usage"]["total_tokens"]
                return obj_list, token
            
            elif response.status == 429:
                raise CustomError("Rate limit reached.") 
            elif response.status == 500:
                raise CustomError("Server error.")
            else:
                response.raise_for_status()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(40),retry=retry_if_exception_type(CustomError))
    async def split_commands(self, session: aiohttp.ClientSession, user_msg: str) -> Union[dict, int]:
        body: dict
        user_msg = user_msg.replace('.', ',')
        preprocessed_msg:str = reduce(lambda s, word: re.sub(r'\b' + word + r'\b', word.upper(), s), ['and', 'while', 'then'], user_msg)
        
        body: dict = self.get_body(model=self.model, user_msg=preprocessed_msg, system_prompt=split_prompt)

        async with session.post(URL, headers=HEADER, data=json.dumps(body)) as response:
            if response.status == 200:
                content: json = await response.json()
                split_and_types: list = json.loads(content["choices"][0]["message"]["content"])
                token: int = content["usage"]["total_tokens"]
                fingerprint: str = content["system_fingerprint"]
                return split_and_types, fingerprint, token
            
            elif response.status == 429:
                raise CustomError("Rate limit reached.") 
            else:
                response.raise_for_status()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(40),retry=retry_if_exception_type(CustomError))
    async def get_inference(self, session:aiohttp.ClientSession, split_cmd: list[str], 
                            dynamic_list: list[str]) -> Union[dict, int]:
        features: list = ["Z-cartesian increase", "Z-cartesian decrease", "Y-cartesian increase", "Y-cartesian decrease", "X-cartesian increase", "X-cartesian decrease"] + dynamic_list

        body: dict = self.get_body(model=self.model, user_msg=f"{split_cmd}\nDYNAMIC FEATURES: {features}", system_prompt=enhanced_inference_prompt)

        async with session.post(URL, headers=HEADER, data=json.dumps(body)) as response:
            if response.status == 200:
                content: json = await response.json()
                fingerprint: str = content["system_fingerprint"]
                classification_w_scores = json.loads(content["choices"][0]["message"]["content"])
                token = content["usage"]["total_tokens"]
                return classification_w_scores, fingerprint, token
            
            elif response.status == 429:
                raise CustomError("Rate limit reached.") 
            elif response.status == 500:
                raise CustomError("Server error.")
            else:
                response.raise_for_status() 
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(40),retry=retry_if_exception_type(CustomError))
    async def get_intensity(self, session:aiohttp.ClientSession, split_cmd: list[str]) -> Union[dict, int]:
        body: dict = self.get_body(model=self.model, user_msg=f"{split_cmd}", system_prompt=intensity_prompt)

        async with session.post(URL, headers=HEADER, data=json.dumps(body)) as response:
            if response.status == 200:
                content: json = await response.json()
                intensities = json.loads(content["choices"][0]["message"]["content"])
                token = content["usage"]["total_tokens"]
                return intensities, token
            
            elif response.status == 429:
                raise CustomError("Rate limit reached.") 
            else:
                response.raise_for_status()    

async def generate_class_list(task: List[str]) -> List[str]:
    return [(x, y) for x, y in zip(task[0]['output'], task[0]['confidence'])]

async def generate_intensity_list(split_task: List[str], intensity_task: List[str], 
                                  type_task:List[str]) -> List[str]:
    return [(x, y, z) for x, y, z in zip(split_task[0]["split"], intensity_task[0]['intensity'], type_task)]


if __name__ == '__main__':
    os.system('clear')
    print("UNCOMMENT THE CODE IN THIS FILE TO RUN THE ASYNC ENDPOINT")

    async def main():  
        start = time.perf_counter()
        usr_cmd="move down and closer to the plate and the microwave"
        gpt: object = GPT(model="gpt-4-1106-preview") 
        total_tokens: int = 0 
        async with aiohttp.ClientSession() as session:
            tasks = [
                        asyncio.create_task(gpt.get_obj_list(session=session, user_msg=usr_cmd)),
                        asyncio.create_task(gpt.split_commands(session=session, user_msg=usr_cmd))
                    ]
            task1, task2 = await asyncio.gather(*tasks)
            object_list = task1[0]
            split_cmd = task2[0]["split"] 
            trajectory_type = task2[0]["type"] 
            total_tokens += (task1[-1] + task2[-1])
            fingerprint_split = task2[1]
            print(f"FINGERPRINT SPLIT: {fingerprint_split}")

            dynamic_features = await gpt.get_dynamic_features(obj_list=object_list, trajectory_list=trajectory_type)

        async with aiohttp.ClientSession() as session:
            tasks = [
                        asyncio.create_task(gpt.get_inference(session=session, split_cmd=split_cmd, dynamic_list=dynamic_features)),
                        asyncio.create_task(gpt.get_intensity(session=session, split_cmd=split_cmd))
                    ]
            task3, task4 = await asyncio.gather(*tasks)
            classification = task3[0]["output"]
            total_tokens += (task3[-1] + task4[-1])
            fingerprint_inference = task3[1]
            print(f"FINGERPRINT INFERENCE: {fingerprint_inference}")

            subtasks = [
                        asyncio.create_task(generate_class_list(task=task3)),
                        asyncio.create_task(generate_intensity_list(split_task=task2, intensity_task=task4, type_task=trajectory_type))
                       ]
            
            class_list, intensity_list = await asyncio.gather(*subtasks)
            

        end = time.perf_counter()
        print(f"OBJECT LIST: {object_list}")
        print(f"DYNAMIC FEATURE: {dynamic_features}")
        print(f"SPLIT COMMAND: {split_cmd}")
        print(f"TRAJECTORY TYPE: {trajectory_type}")
        print(f"CLASSIFICATION: {classification}")
        print(f"CLASS LIST: {class_list}")
        print(f"INTENSITY LIST: {intensity_list}")
        print()
        print(f"TOKEN: {total_tokens}")
        print(f"Finished in {end - start:0.2f} seconds")
        if fingerprint_inference == fingerprint_split:
            print("FINGERPRINT MATCHES")
        
    asyncio.run(main())  
