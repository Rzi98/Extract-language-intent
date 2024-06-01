import sys
import os
import json
import asyncio
import aiohttp 
import yaml
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from collections import deque
from typing import List, Literal, Optional, Union, Dict
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt.embeddings.helper import load_mapper, get_all_paths
from gpt.module.model import GPT

load_dotenv()

chatModel = Literal["gpt-3.5-turbo", "gpt-4"]
embedModel = Literal["text-embedding-ada-002", "text-embedding-3-large"]
embeddingModel = "text-embedding-ada-002" ### CHANGE
# embeddingModel = "text-embedding-3-large" ### CHANGE

CONFIG_PATH: Path = './gpt/config/config.yaml'
TOKEN = os.getenv('OPENAI_API_KEY')
URL = "https://api.openai.com/v1/embeddings"

STATIC_EMBED_PATH = get_all_paths(config_path=CONFIG_PATH)['STATIC_EMBED_PATH']
DYNAMIC_EMBED_PATH = get_all_paths(config_path=CONFIG_PATH)['DYNAMIC_EMBED_PATH']

def save_embeddings(path:Path, data:json) -> None:
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=4)
    print("Saved static mapper successfully!")

def static_embedding_exists(static_embed_path:Path) -> bool:
    """
        check if the static mapper with embeddings already exists
    """
    try:
        with open(file=static_embed_path, mode='r') as json_file:
            data = json.load(json_file)
        if data:
            return True
        return False
    
    except FileNotFoundError:
        return False

async def get_static_embeddings(fix_map:deque,
                          path:Optional[Path]=STATIC_EMBED_PATH,
                          ):
    """
        Fixed embeddings for all test cases
        Returns:
            static_embeddings (dict): {cmd: {embeddings, patterns}}
    """
    if not static_embedding_exists(static_embed_path=path):
        static_embeddings = await MapperEmbeddings.embed_staticmap(static_map=deque(fix_map))
        save_embeddings(path=path, data=static_embeddings)
    else:
        static_embeddings = load_mapper(mapper_path=path, 
                                        STATIC_EMBED_PATH=STATIC_EMBED_PATH, 
                                        DYNAMIC_EMBED_PATH=DYNAMIC_EMBED_PATH)
        print("Static VectorDB already exists!")
    return static_embeddings    

async def compare_embeddings(test_dict: dict, ref_dict: dict, obj_list: list, dynamic_map: dict):
    prediction: dict = {}
    tasks: list = []

    for i, test_cmd in enumerate(test_dict):
        print("-"*75)
        print(f"Running segment: {i + 1}")
        print(f"Test cmd: {test_cmd}")

        if not any(word in test_cmd.split() for word in obj_list):  # STATIC EMBEDDING
            test_embeddings = test_dict[test_cmd]['embeddings']
            max_similarity_score: float = -1
            closest_pattern: str = None
            closest_feature: str = None

            for feature in ref_dict:
                for ref_cmd, ref_embeddings in ref_dict[feature].items():
                    similarity_score = cosine_similarity([test_embeddings], [ref_embeddings])
                    if similarity_score > max_similarity_score:
                        max_similarity_score = similarity_score
                        closest_pattern = ref_cmd
                        closest_feature = feature

            if closest_pattern and max_similarity_score >= 0:  
                prediction[test_cmd] = {'closest_pattern': closest_pattern,
                                        'predicted_feature': closest_feature,
                                        'confidence_score': round(max_similarity_score[0][0], 5)}
            else:
                print(f"No closest pattern found for: {test_cmd}")
        else:  # DYNAMIC EMBEDDING
            print("dynamic embeddings")
            task = asyncio.create_task(
                MapperEmbeddings.embed_dynamicmap(dynamic_deq=deque(dynamic_map),
                                                  single_cmd=test_cmd,
                                                  obj_list=obj_list,
                                                  ))
            tasks.append(task)

    dynamic_results = await asyncio.gather(*tasks)

    for dynamic_prediction in dynamic_results:
        prediction.update(dynamic_prediction)

    return prediction

class TestEmbedding(GPT):

    def __init__(self, id: int, yaml_path: Optional[Path] = "./gpt/config/config.yaml") -> None:
        
        self.id = id
        self.object_path, self.split_path = self.load_yaml(yaml_path=yaml_path)

    def load_yaml(self, yaml_path:Path) -> Union[Path,Path]:
        with open(file=yaml_path, mode='r') as f:
            config = yaml.safe_load(f)  
            return config['OBJ_LIST'], config['SPLITS']

    async def load_objects(self, id: int) -> List[str]:
        with open(file=self.object_path, mode='r') as f:
            obj_list = json.load(f)
        return obj_list["obj_list"][str(id)]
    
    async def load_splits(self, id: int) -> List[str]: 
        with open(file=self.split_path, mode='r') as f:
            split_cmds = json.load(f)
        return split_cmds["split_cmds"][str(id)]
    
    async def loads(self)-> asyncio.coroutines:
        tasks = [
                    self.load_objects(id=self.id), 
                    self.load_splits(id=self.id)
                ]
        return await asyncio.gather(*tasks) 
    
    async def post_embeddings(self, split_cmd: list) -> dict:
        embedding_url = URL  
        headers = {  
            "Authorization": f"Bearer {TOKEN}",  
            "Content-Type": "application/json"  
        }  
        embed_lookup:dict = {}  
        token_count:int = 0  

        async with aiohttp.ClientSession() as session:  
            for cmd in split_cmd:  
                response = await self.post(session=session, url=embedding_url, headers=headers, body=cmd)  
                inner_lookup = {'embeddings': response['data'][0]['embedding']}  
                token_count += response['usage']['total_tokens']  
                embed_lookup[cmd] = inner_lookup  
  
        return embed_lookup, token_count
    
    async def post(self, session:aiohttp.ClientSession, url:str, headers:dict, body:dict):  
        data = {  
            "input": body,  
            "model": embeddingModel,  
            "encoding_format": "float",  
        }  

        retries: int = 0
        max_retries: int = 3

        while True:
            async with session.post(url, headers=headers, data=json.dumps(data)) as resp:
                try:
                    if resp.status == 200:  
                        return await resp.json()
                    else:
                        raise Exception(f"Unexpected response status: {resp.status}")

                except Exception as e:
                    print(f"ERROR: {e}")
                    print("Retrying...")
                    await asyncio.sleep(20 * retries)
                    retries += 1
                    if retries > max_retries:
                        raise Exception(f"API call failed after maximum retries. Last exception: {str(e)}")

class MapperEmbeddings:
    """
        This class is used to embed the {{benchmark corpus}} into a vector space.
    """
    @staticmethod
    async def post_embed(session: aiohttp.ClientSession, text:str) -> list:  
        embedding_url = URL  
        headers = {  
            "Authorization": f"Bearer {TOKEN}",  
            "Content-Type": "application/json"  
            }  
          
        data = {  
            "input": text,  
            "model": embeddingModel,   ### Change Model Name ###
            "encoding_format": "float",  
            }  
        
        retries: int = 0
        max_retries: int = 3

        while True:
            try:
                async with session.post(embedding_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['data'][0]['embedding']
                    else:
                        raise Exception(f"Unexpected response status: {response.status}")
            except Exception as e:
                print(f"ERROR: {e}")
                print("Retrying...")
                retries += 1
                await asyncio.sleep(20 * retries)
                if retries > max_retries:
                    raise Exception(f"API call failed after maximum retries. Last exception: {str(e)}")
    
    @classmethod  
    async def embed_staticmap(cls, static_map:deque) -> list:
        """ Get the embedding of the cartesian features """  
        async with aiohttp.ClientSession() as session:    
            new_static_map:dict = {}  
            for item in static_map:  
                inner = {}  
                tasks = []  
                for pattern in item['patterns']:  
                    task: asyncio.tasks = asyncio.create_task(cls.post_embed(session, pattern))  
                    tasks.append(task)  
                embeddings: list[float] = await asyncio.gather(*tasks)  
                for pattern, embedding in zip(item['patterns'], embeddings):  
                    inner[pattern] = embedding  
                new_static_map[item['feature']] = inner  
            return new_static_map
        
    @classmethod
    async def sentence_similarity(cls, text_A:str, text_B:str) -> float:
        async with aiohttp.ClientSession() as session:
            tasks: asyncio.tasks = [
                asyncio.create_task(cls.post_embed(session=session,text=text_A)),
                asyncio.create_task(cls.post_embed(session=session,text=text_B))
            ]
            embedding1, embedding2 = await asyncio.gather(*tasks)
            similarity = cosine_similarity([embedding1], [embedding2])
            return round(similarity[0][0], 5)
        
    @classmethod
    async def compare_all_objects(cls, specific_deque: deque, test_cmd: str):
        max_similarity_score: int = -1
        closest_pattern: str = None
        closest_feature: str = None
        prediction: Dict[str, dict] = {}
        tasks: List[asyncio.Task] = []

        async def process_feature(feature:str, patterns:List[str], test_cmd:str):
            nonlocal max_similarity_score, closest_pattern, closest_feature

            for pattern in patterns['patterns']:
                similarity_score = await cls.sentence_similarity(text_A=test_cmd, text_B=pattern)
                if similarity_score > max_similarity_score:
                    max_similarity_score = similarity_score
                    closest_pattern = pattern
                    closest_feature = feature

        for _, commands in specific_deque.items():
            for feature, patterns in commands.items():
                tasks.append(process_feature(feature=feature, 
                                             patterns=patterns, 
                                             test_cmd=test_cmd))

        await asyncio.gather(*tasks)

        prediction[test_cmd] = {
            'closest_pattern': closest_pattern,
            'predicted_feature': closest_feature,
            'confidence_score': max_similarity_score
        } 

        return prediction 
    
    @classmethod
    def embed_dynamicmap(cls, dynamic_deq:deque, single_cmd:str, obj_list: Union[list[str],str]) -> list:
        """
            ALL OBJECT APPROACH
        """
        specific_deque = {
                            obj: {
                                item['feature'].replace('{obj}', obj): {
                                    'patterns': [pattern.replace('{obj}', obj) for pattern in item['patterns']],
                                }
                                for item in dynamic_deq
                            }
                            for obj in obj_list
                        }

        return cls.compare_all_objects(specific_deque=specific_deque, test_cmd=single_cmd)


class MapperBestObjEmbeddings(MapperEmbeddings):

    @staticmethod
    def embed_object(single_cmd:str, obj_list: Union[list[str], str]):
        """  
            BEST OBJECT APPROACH: 
            Get Best object -> Put selected object in dynamic map -> Embed the dynamic map -> Compare the embeddings

            Use sentence similarity to compare the object with 
            the user_cmd embeddings to find the closest object.
        """
        scores = []
        for obj in obj_list:
            similarity_score = MapperEmbeddings.sentence_similarity(object_name=obj, single_cmd=single_cmd)
            scores.append((obj, similarity_score))

        best_match_obj_w_score = (max(scores, key=lambda x: x[1])[0], max(scores, key=lambda x: x[1])[1])
        best_obj = best_match_obj_w_score[0]
        
        return {single_cmd : best_obj}
    
    @staticmethod
    def compare_dynamic_embeddings(specific_deque:deque, single_cmd_w_obj:str):
        """
            BEST Object Approach:
            This method is used to compare the user_cmd with the dynamic map.
            The dynamic map is a deque of specific items {features, patterns}
        """
        print("Comparing dynamic embeddings...")
        max_similarity_score = -1  
        closest_pattern = None

        score_logs, prediction = {}, {}
        for feature in specific_deque:
            scores = []
            for pattern in specific_deque[feature]['patterns']:
                similarity_score = MapperEmbeddings.sentence_similarity(text_A=pattern, text_B=single_cmd_w_obj)
                scores.append((pattern, similarity_score))
                if similarity_score > max_similarity_score:
                        max_similarity_score = similarity_score
                        closest_pattern = pattern
                        closest_feature = feature
            
            if single_cmd_w_obj in score_logs:
                score_logs[single_cmd_w_obj].extend(scores)
            else:
                score_logs[single_cmd_w_obj] = scores
        
        prediction[single_cmd_w_obj] = {'closest_pattern': closest_pattern, 
                                        'predicted_feature': closest_feature, 
                                        'confidence_score': max_similarity_score}


    @classmethod
    def embed_dynamicmap(cls, dynamic_deq:deque, single_cmd:str, obj_list: Union[list[str],str]) -> list:
        """
            BEST OBJECT APPROACH
        """
        
        single_map_obj = cls.embed_object(single_cmd=single_cmd, obj_list=obj_list)
        print(f"SINGLE MAP: {single_map_obj}")
        specific_deque = {
                    item['feature'].replace('{obj}', list(single_map_obj.values())[0]): {
                        'patterns': [pattern.replace('{obj}', list(single_map_obj.values())[0]) for pattern in item['patterns']],
                        'embeddings': [cls.post_embed(pattern) for pattern in item['patterns']]
                    }
                    for item in dynamic_deq
                 }

        return cls.compare_dynamic_embeddings(specific_deque=specific_deque, single_cmd_w_obj=list(single_map_obj.keys())[0])



async def main():
    embed = TestEmbedding(id=1)
    print(split_cmd)
    object_list, split_cmd = await embed.loads()
    embed_dict, token = await embed.post_embeddings(split_cmd=split_cmd)

    print(f"OBJECT LIST: {object_list}")
    print(f"SPLIT CMD: {split_cmd}")
    print(f"EMBEDDINGS: {embed_dict}")
    print(f"TOKEN COUNT: {token}")

    score = await MapperEmbeddings.sentence_similarity(object_name="Z-cartesian increase", single_cmd="Z-cartesian increase")
    print(f"score: {score}")
if __name__ == '__main__':
    os.system('clear')
    
    # pass
    # asyncio.run(main())
    