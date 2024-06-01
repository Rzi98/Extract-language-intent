import asyncio
import yaml
import aiohttp
import time
import os
import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import List

from gpt.module.prompts import *
from gpt.module.endpoints import GPT, generate_class_list, generate_intensity_list
from gpt.module.helpers import clear_res_files
from gpt.module.data_handler import DataProcessor, SingleDataProcessor, mode_type
from gpt.module.inference_checkers import inference_checker
from gpt.module.datatypes import ConfigArgument, InputPath, OutputPath, Correction

CONFIG_PATH = './gpt/config/config.yaml'
BATCH_PATH = './gpt/config/batch.yaml'


def read_config(filepath: Path):
    with open(file=filepath, mode='r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def setup_config(CONFIG_PATH: Path = CONFIG_PATH, BATCH_PATH: Path = BATCH_PATH):
    os.system('clear')

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4-1106-preview', help='Model to use for inference', action='store')
    parser.add_argument('--batch', type=int, help='Batch of data for inference', action='store', choices=[1,2,3,4,5,9], nargs='?', required=True)
    parser.add_argument('--reset',  help='Delete result folder', action='store_true', required=False, default=False)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    config: dict = read_config(filepath=CONFIG_PATH)
    data_path: Path = config['DATA']

    batch_config: dict = read_config(filepath=BATCH_PATH)
    batches: dict = batch_config['batches']

    if args.model == 'main':
        MODEL = config['MODELS']['MAIN']
    elif args.model == 'gpt-4':
        MODEL = config['MODELS']['CHAT']['GPT4']
    elif args.model == 'gpt-3':
        MODEL = config['MODELS']['CHAT']['GPT3']
    else:
        MODEL = config['MODELS']['MAIN']

    result_path: Path = os.path.join((config['RESULT']), MODEL)
    logger_filepath: Path = os.path.join(result_path, f'{MODEL}.log')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if args.reset:
        clear_res_files(result_path)

    file_handler = logging.FileHandler(logger_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print("CURRENTLY RUNNING ...")
    print(f"MODEL: {MODEL}")
    print(f"RESULT PATH: {result_path}")
    print(f"LOGGER FILEPATH: {logger_filepath}", end='\n\n')
    print(f"CURRENTLY RUNNING BATCH {args.batch}")

    time.sleep(2)
    os.system('clear')

    return ConfigArgument(model=MODEL, batch_no=args.batch, batches=batches), InputPath(data=data_path),\
        OutputPath(logger=logger_filepath, result=result_path), logger


async def main():  
    os.system('clear')
    config: ConfigArgument
    input_path: InputPath
    output_path: OutputPath
    logger: logging.Logger
    
    config, input_path, output_path, logger = setup_config(CONFIG_PATH=CONFIG_PATH, BATCH_PATH=BATCH_PATH)

    data_processor: object = DataProcessor()
    data = data_processor.load_data(input_path.data, config.batches[config.batch_no])

    class_scores_processor = SingleDataProcessor(mode='classification_w_scores')
    intensity_types_processor = SingleDataProcessor(mode='intensity_w_types') 
    split_processor = SingleDataProcessor(mode='split_cmds')
    obj_processor = SingleDataProcessor(mode='obj_list')

    mode1: mode_type = 'objects'
    mode2: mode_type = 'split'
    mode3: mode_type = 'classify'
    mode4: mode_type = 'token_count'
    mode5: mode_type = 'timer'
    mode6: mode_type = 'classification_w_scores'
    mode7: mode_type = 'intensity_w_types'
    mode8: mode_type = 'split_cmds'
    mode9: mode_type = 'obj_list'

    for usr_cmd in data:
        start = time.perf_counter()
        gpt: GPT = GPT(model="gpt-4-1106-preview")
        total_tokens: int = 0
        id: int = data[usr_cmd]['id']

        async with aiohttp.ClientSession() as session:
            tasks = [
                        asyncio.create_task(gpt.get_obj_list(session=session, user_msg=usr_cmd)),
                        asyncio.create_task(gpt.split_commands(session=session, user_msg=usr_cmd))
                    ]
            task1, task2 = await asyncio.gather(*tasks)
            object_list: List[str] = task1[0]
            split_cmd: List[str] = task2[0]["split"] 
            trajectory_type: List[str] = task2[0]["type"] 
            total_tokens += (task1[-1] + task2[-1])
            fingerprint_split = task2[1]

            dynamic_features = await gpt.get_dynamic_features(obj_list=object_list, trajectory_list=trajectory_type)

        async with aiohttp.ClientSession() as session:
            tasks = [
                        asyncio.create_task(gpt.get_inference(session=session, split_cmd=split_cmd, dynamic_list=dynamic_features)),
                        asyncio.create_task(gpt.get_intensity(session=session, split_cmd=split_cmd))
                    ]
            
            task3, task4 = await asyncio.gather(*tasks)
            classification: List[str] = task3[0]["output"]
            total_tokens += (task3[-1] + task4[-1])
            fingerprint_inference = task3[1]

            subtasks = [
                        asyncio.create_task(generate_class_list(task=task3)),
                        asyncio.create_task(generate_intensity_list(split_task=task2, intensity_task=task4, type_task=trajectory_type))
                       ]
            
            class_list: List[str]
            intensity_list: List[str]

            class_list, intensity_list = await asyncio.gather(*subtasks)
            
        end: float = time.perf_counter()
        
        logger.info(f"ID: {id}")
        logger.info(f"INPUT: '{usr_cmd}'")
        logger.info(f"OBJECT LIST: {object_list}")
        logger.info(f"DYNAMIC FEATURE: {dynamic_features}")
        logger.info(f"SPLIT COMMAND: {split_cmd}")
        logger.info(f"TRAJECTORY TYPE: {trajectory_type}")
        logger.info(f"CLASSIFICATION: {classification}")
        logger.info(f"CLASS LIST: {class_list}")
        logger.info(f"INTENSITY LIST: {intensity_list}")
        logger.info(f"TOKEN: {total_tokens}")
        logger.info(f"FINGERPRINT SPLIT: {fingerprint_split}")
        logger.info(f"FINGERPRINT INFERENCE: {fingerprint_inference}")
        if fingerprint_inference == fingerprint_split:
            logger.info("MATCH!!!")
        else:
            logger.info("NOT MATCH!!!")
        logger.info(f"Finished in {end - start:0.2f} seconds")

        print(f"ID: {id}")
        print(f"INPUT: '{usr_cmd}'")
        print(f"OBJECT LIST: {object_list}")
        print(f"DYNAMIC FEATURE: {dynamic_features}")
        print(f"SPLIT COMMAND: {split_cmd}")
        print(f"TRAJECTORY TYPE: {trajectory_type}")
        print(f"CLASSIFICATION: {classification}")
        print(f"CLASS LIST: {class_list}")
        print(f"INTENSITY LIST: {intensity_list}")
        print(f"TOKEN: {total_tokens}")
        print(f"Finished in {end - start:0.2f} seconds")

        predicted_data: dict = {
                            key: value
                            for key, value in zip(
                                ['inference_objects', 'inference_splits', 'inference_class', 'total_tokens', 'exec_time'],
                                [object_list, split_cmd, classification, total_tokens,f"{end - start: 0.2f}"]
                            )
                         }
        
        object_correction: Correction
        split_correction: Correction
        class_correction: Correction

        object_correction, split_correction, class_correction = await inference_checker(data[usr_cmd], predicted_data)

        data_processor.store_data(id, object_correction, split_correction, class_correction,
                                   predicted_data['total_tokens'], predicted_data['exec_time'])
        
        class_scores_processor.store_data(id,class_list)
        intensity_types_processor.store_data(id,intensity_list)
        split_processor.store_data(id,split_cmd)
        obj_processor.store_data(id,object_list)
        
        correction_path = os.path.join(output_path.result,'correction')
        prediction_path = os.path.join(output_path.result,'prediction')

        if not os.path.exists(correction_path) or not os.path.exists(prediction_path):
            os.makedirs(correction_path)
            os.makedirs(prediction_path)
                
        tasks = [
                    asyncio.create_task(data_processor.save_to_json(filename=os.path.join(correction_path,mode1+'.json'), mode=mode1)),
                    asyncio.create_task(data_processor.save_to_json(filename=os.path.join(correction_path,mode2+'.json'), mode=mode2)),
                    asyncio.create_task(data_processor.save_to_json(filename=os.path.join(correction_path,mode3+'.json'), mode=mode3)),
                    asyncio.create_task(data_processor.save_to_json(filename=os.path.join(prediction_path,mode4+'.json'), mode=mode4)),
                    asyncio.create_task(data_processor.save_to_json(filename=os.path.join(prediction_path,mode5+'.json'), mode=mode5)),
                    asyncio.create_task(class_scores_processor.save_to_json(filename=os.path.join(prediction_path,mode6+'.json'), mode=mode6)),
                    asyncio.create_task(intensity_types_processor.save_to_json(filename=os.path.join(prediction_path,mode7+'.json'), mode=mode7)),
                    asyncio.create_task(split_processor.save_to_json(filename=os.path.join(prediction_path,mode8+'.json'), mode=mode8)),
                    asyncio.create_task(obj_processor.save_to_json(filename=os.path.join(prediction_path,mode9+'.json'), mode=mode9))
                ]
        
        await asyncio.gather(*tasks)
    
    os.system('clear')
    print("COMPLETED")
    print(f"### Batch no: {config.batch_no} | Range: {config.batches[config.batch_no]} ###")

if __name__ == '__main__':
    asyncio.run(main())