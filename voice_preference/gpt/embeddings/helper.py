import json
import os
from collections import deque
import yaml
from pathlib import Path

def get_all_paths(config_path:Path):

    global MAPPER_PATH, STATIC_EMBED_PATH, DYNAMIC_EMBED_PATH
    global generic_path, result_predict_path, logger_path

    with open(file=config_path, mode='r') as fp:
        config = yaml.safe_load(fp)
    
    MAPPER_PATH = config['MAPPER_PATH']
    STATIC_EMBED_PATH = config['STATIC_EMBED_PATH']
    DYNAMIC_EMBED_PATH = config['DYNAMIC_EMBED_PATH']

    generic_path = config['generic_path']
    result_predict_path = config['result_predict_path']
    try:
        logger_path = config['log_path']
    except:
        logger_path = None

    path_lookup = {
        'MAPPER_PATH': MAPPER_PATH,
        'STATIC_EMBED_PATH': STATIC_EMBED_PATH,
        'DYNAMIC_EMBED_PATH': DYNAMIC_EMBED_PATH,
        'generic_path': generic_path,
        'result_predict_path': result_predict_path,
        'logger_path': logger_path
    }

    return path_lookup
        

def load_mapper(mapper_path:Path, STATIC_EMBED_PATH: Path, DYNAMIC_EMBED_PATH: Path):
    """
        load and separate the mapper into static and dynamic
    """
    
    with open(mapper_path, 'r') as fp:
        data = json.load(fp)
    if mapper_path == STATIC_EMBED_PATH or mapper_path == DYNAMIC_EMBED_PATH:
        return data
    # FOR MAPPER
    mapper = deque(data['intents'])
    dynamic_mapper = [mapper.popleft() for _ in range(2)]
    return mapper, dynamic_mapper

def load_testcases(path:Path):
    """
        Load test data from 100 dataset
    """
    with open(file=path, mode='r') as f:
        data = json.load(f)
    return data

def get_test_objects(obj_list:list):
    """
        Get the list of objects selected for the testing
    """
    obj_path: Path = './gpt/data/object.json'
    if len(obj_list) < 4:
        with open(file=obj_path, mode='r') as f:
            data = json.load(f)    
            for obj in data['objects']:
                if obj not in obj_list:
                    obj_list.append(obj)
                    if len(obj_list) == 4:
                        break
    return obj_list


if __name__ == '__main__':
    pass


    
