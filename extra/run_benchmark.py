from argparse import ArgumentParser
import os, json, itertools, logging, time
from pathlib import Path

from module.datatypes import Data, SimilarityObjResponse, IntensityTypeResponse, direction
from module.gpt import Model
from module.process import Processing, FeaturesLabels

# parser = ArgumentParser()
# parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model to use for inference', action='store')
# parser.add_argument('-s', '--start', type=int, help='Start index for inference', action='store', nargs='?', required=False)
# parser.add_argument('-e', '--end', type=int, help='End index for inference', action='store', nargs='?', required=False)
# parser.add_argument('-r', '--reset', help='Clear result folder', action='store_true', required=False, default=False)
# args = parser.parse_args()

load_path = './data/data.json'
# save_path = f'../result/{args.model}/output.json'
# index_log_path = f'../result/{args.model}/index.log'
# verbose_log_path = f'../result/{args.model}/verbose.log'

# logger1 = logging.getLogger('logger1')
# logger1.setLevel(logging.INFO) 
# file_handler1 = logging.FileHandler(index_log_path)
# formatter1 = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# file_handler1.setFormatter(formatter1)
# logger1.addHandler(file_handler1)

# logger2 = logging.getLogger('logger2')
# logger2.setLevel(logging.INFO)  # Set the logger's level to INFO
# file_handler2 = logging.FileHandler(verbose_log_path)
# formatter2 = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# file_handler2.setFormatter(formatter2)
# logger2.addHandler(file_handler2)

def check_existing_json(save_path: Path) -> dict:
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                existing_lookup = json.load(f)
        else:
            existing_lookup = {}
        
        return existing_lookup

if __name__ == '__main__':
    os.system('clear')
    
    # if args.reset and os.path.exists(save_path):
    #     os.remove(save_path)
    #     os.remove(index_log_path)
    #     os.remove(verbose_log_path)
    #     print("Result files deleted")
    #     time.sleep(2)
    #     os.system('clear')
    
    # print("Running MODEL: ", args.model)
    # print("START: ", args.start)
    # print("END: ", args.end)
    time.sleep(2)
    os.system('clear')
    
    with open(file=load_path) as f:
        data:json = json.load(f)
    
    
    # logger1.info(f"START: {args.start}")
    # logger1.info(f"END: {args.end}")
    # logger1.info(f"NEXT START: {args.end + 1}")
    # logger1.info(f"="*150)
    # sliced_data = dict(itertools.islice(data.items(), args.start-1, args.end))
    sliced_data = dict(itertools.islice(data.items(), 0, 2))
    test_cases = [
        Data(id=int(k)+1,
             corrections=v['correction'], 
             obj_list=v['obj_list'],
             gt_type=v['gt_type'],
             gt_target_obj=v['gt_target_object'],
             gt_direction=v['gt_direction'],
             gt_intensity=v['gt_intensity'],
             gt_cart_axes=v['gt_cart_axes'],
             gt_split_commands=v['gt_split'],
             gt_classification=v['gt_classification']) 
        for k,v in sliced_data.items()]
    
    print(test_cases)
    
    # lookup = check_existing_json(save_path=save_path)
    # for test_data in test_cases:

    #     test_msg, test_id = Model.data_to_user_input(test_data)

    #     m = Model(model=args.model)

    #     sim_dict: SimilarityObjResponse = m.get_closet_obj(test_msg)
    #     intensity_dict: IntensityTypeResponse = m.get_intensity_type(test_msg)
    #     predicted_dict = Processing.dict_to_PredictData(id=test_id, sim_dict=sim_dict, intensity_dict=intensity_dict).model_dump()
    #     fl = FeaturesLabels(id=predicted_dict['id'], similarity=predicted_dict['similarity'],
    #                         closest_obj=predicted_dict['closest_obj'], intensity=predicted_dict['intensity'], 
    #                         type=predicted_dict['type'])
    #     result = Processing.classify(user_input=test_msg, data=fl, model=args.model).model_dump()
    #     index = result.pop('id')
    #     lookup[index] = result
        
    #     logger2.info(f"INDEX #{index}")
    #     print(f"INDEX #{index}")
    #     for k,v in result.items():
    #         logger2.info(f"{k}: {v}")
    #         print(f"{k}: {v}")
    #     print("="*150)
    #     logger2.info("="*150)

    #     with open(save_path, 'w') as f:
    #         json.dump(lookup, f, indent=4)

    # print(lookup)

    # for test_data in test_cases:
    #     test_msg, test_id = Model.data_to_user_input(data=test_data)