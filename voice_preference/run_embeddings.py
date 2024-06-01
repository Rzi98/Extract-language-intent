import openai
import json
import os
import time
import asyncio
import logging
from dotenv import load_dotenv
from argparse import ArgumentParser
from pathlib import Path
from itertools import islice
from collections import OrderedDict

from gpt.module.inference_checkers import check_classify
from gpt.embeddings.models_embed import (TestEmbedding, get_static_embeddings, 
                                         compare_embeddings)
from gpt.embeddings.helper import (load_mapper,load_testcases,
                    get_test_objects,get_all_paths)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CONFIG_PATH: Path = './gpt/config/config.yaml'
BATCH_PATH = './gpt/config/batch.yaml'

batch_r1 = (1, 20) 
batch_r2 = (21, 40) 
batch_r3 = (41, 60)
batch_r4 = (61, 80)
batch_r5 = (81, 100)

batch_temp = (2,3) # FOR TESTING PURPOSES

batches = {
        1 : batch_r1,
        2 : batch_r2,
        3 : batch_r3,
        4 : batch_r4,
        5 : batch_r5,
        9 : batch_temp
        }

def setup_argparse():
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, help='Batch of data for inference', action='store', choices=[1,2,3,4,5,9], nargs='?', required=True)
    parser.add_argument('-r', '--reset', help='Clear result folder', action='store_true', required=False, default=False)
    args = parser.parse_args()
    return args

def setup_logger(file: Path):
    if not os.path.exists(file):
        os.makedirs(file.rsplit('/', 1)[0], exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    file_handler = logging.FileHandler(file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def print_and_log(logger:logging.Logger, id:int, full_cmd:str, 
                    objects_extracted:list, split_cmd:list, 
                    obj_list:list, predict:dict) -> None:
        
    logger.info("="*150)
    logger.info(f"ID: {id}")
    logger.info(f"FULL COMMANDS: {full_cmd}")
    logger.info(f"OBJECT EXTRACTED: {objects_extracted}")
    logger.info(f"SPLIT COMMANDS: {split_cmd}")
    logger.info(f"OBJECT LIST PARSE: {obj_list}")
    logger.info(f"RESULT: {predict}")
    logger.info(f"ELAPSED TIME: {predict['elapsed_time']}s")
    logger.info(f"CLASS CORRECTION: {predict['correction']}")

    print("="*150)
    print(f"ID: {id}")
    print(f"FULL COMMANDS: {full_cmd}")
    print(f"OBJECT EXTRACTED: {objects_extracted}")
    print(f"SPLIT COMMANDS: {split_cmd}")
    print(f"OBJECT LIST PARSE: {obj_list}")
    print(f"RESULT: {predict}")
    print(f"ELAPSED TIME: {predict['elapsed_time']}s")
    print(f"CLASS CORRECTION: {predict['correction']}")
    print()

def save_result(predict: dict, full_predict_logs: dict, result_predict_path: Path, id: str):
    id_str = str(id)

    full_predict_logs[id_str] = predict
    
    if os.path.exists(result_predict_path):
        with open(file=result_predict_path, mode='r') as f:
            existing_predict_logs = json.load(f)
    else:
        os.makedirs(result_predict_path.rsplit('/', 1)[0], exist_ok=True)
        existing_predict_logs = {}

    existing_predict_logs.update(full_predict_logs)

    with open(result_predict_path, 'w') as f:
        json.dump(existing_predict_logs, f, indent=4)

async def main():
    MAPPER_PATH: Path
    STATIC_EMBED_PATH: Path 
    DYNAMIC_EMBED_PATH: Path 
    generic_path: Path 
    result_predict_path: Path 

    path_lookup = get_all_paths(config_path=CONFIG_PATH)
    

    MAPPER_PATH = path_lookup['MAPPER_PATH']
    STATIC_EMBED_PATH = path_lookup['STATIC_EMBED_PATH']
    DYNAMIC_EMBED_PATH = path_lookup['DYNAMIC_EMBED_PATH']
    generic_path = path_lookup['generic_path']
    result_predict_path = path_lookup['result_predict_path']
    logger_path = path_lookup['logger_path']

    args = setup_argparse()

    if args.reset and os.path.exists(result_predict_path):
        print("CLEARING RESULT FOLDER ...")
        time.sleep(2)
        if os.path.exists(result_predict_path):
            os.remove(path=result_predict_path)
            print("Cleared result file.")
        if os.path.exists(logger_path):
            os.remove(path=logger_path)
            print("Cleared logger file.")

    logger = setup_logger(file=logger_path)

    batch = batches[args.batch]
    print("CURRENTLY RUNNING ...")
    print(f"BATCH: {batch}")
    time.sleep(2)
    os.system('clear')

    static_map, dynamic_map = load_mapper(mapper_path=MAPPER_PATH, 
                                          STATIC_EMBED_PATH=STATIC_EMBED_PATH, 
                                          DYNAMIC_EMBED_PATH=DYNAMIC_EMBED_PATH)
    static_embeddings = await get_static_embeddings(fix_map=static_map, path=STATIC_EMBED_PATH)
    data = load_testcases(path=generic_path)
    
    full_predict_logs: dict = {}
    for full_cmd in dict(islice(data.items(),batch[0]-1,batch[1])): 
        start = time.perf_counter()
        id = data[full_cmd]['id']
        embed = TestEmbedding(id=id)
        objects_extracted, split_cmd = await embed.loads()
        print(f"split_cmd: {split_cmd}")
        embed_dict, _ = await embed.post_embeddings(split_cmd=split_cmd)
        obj_list = get_test_objects(obj_list=objects_extracted)
        predict = await compare_embeddings(test_dict=embed_dict, 
                                              ref_dict=static_embeddings, 
                                              obj_list=obj_list, 
                                              dynamic_map=dynamic_map)
        predicted_features: list = []  
        ordered_predict: OrderedDict = OrderedDict()  
        
        for cmd in split_cmd:  
            cmd_value = predict.get(cmd, {})  
            ordered_predict[cmd] = cmd_value  
        
            if isinstance(cmd_value, dict) and 'predicted_feature' in cmd_value:  
                predicted_features.append(cmd_value['predicted_feature'])  
        predict: dict = dict(ordered_predict)  
        end = time.perf_counter()
        predict['elapsed_time'] = round(end-start, 5)
        class_correction = await check_classify(
                    src_classify_res=data[full_cmd]['selected_features'], 
                    pred_classify_res=predicted_features
                )
        
        predict['correction'] = class_correction

        print_and_log(logger=logger, id=id, full_cmd=full_cmd, 
                      objects_extracted=objects_extracted, 
                      split_cmd=split_cmd, obj_list=obj_list, 
                      predict=predict)

        save_result(predict=predict, full_predict_logs=full_predict_logs, 
            result_predict_path=result_predict_path, id=id)
            
    os.system('clear')
    print("DONE")
    if args.batch != 5 and args.batch != 9:
        print(f"RUN NEXT: python run_embedding.py -b {args.batch+1}")
    
    else:
        print("ALL BATCHES DONE")

if __name__ == '__main__':
    os.system('clear')
    asyncio.run(main())   