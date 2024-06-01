import json
from itertools import islice
from typing import (List, Literal, Optional, 
                    Tuple)
from pathlib import Path

mode_type = Literal["objects", "split", "classify", "token_count", "timer", "classification_w_scores", "intensity_w_types"]

class DataProcessor:
    """
        Handles the data storage and saving of the results of the GPT API.
    """
    def __init__(self):
        self.reset_storage_data()

    @staticmethod
    def load_data(data_filepath: Path, batch_range: Tuple[int, int]) -> dict:
        with open(data_filepath, 'r') as json_file:
                data = json.load(json_file)
        start_index, end_index = batch_range[0], batch_range[1]
        return dict(islice(data.items(), start_index - 1, end_index))
    
    def reset_storage_data(self):
        self.data = {
            "objects": {},
            "split": {},
            "classify": {},
            "token_count": {},
            "timer": {}
        }

    def store_data(self, id: int, objects: list, 
                   split_cmd: list, class_list: list, 
                   token_count: int, exec_time: float) -> None:
        
        self.data["objects"][id] = objects
        self.data["split"][id] = split_cmd
        self.data["classify"][id] = class_list
        self.data["token_count"][id] = token_count
        self.data["timer"][id] = exec_time

    async def save_to_json(self, filename: Path, mode: Optional[mode_type]='classify') -> None:
        ''' 
        INFERENCE MODE: ['objects', 'split', 'classify', 'token_count', 'timer']
          '''
        
        # Load existing data from the JSON file if it exists
        existing_data = {}
        try:
            with open(filename, 'r') as json_file:
                existing_data = json.load(json_file)
        except FileNotFoundError:
            # print("Generating new json file...")
            pass

        # Check if the mode key exists in existing_data
        if mode not in existing_data:
            existing_data[mode] = {}

        # Update the mode portion of existing_data with the new data
        existing_data[mode].update(self.data[mode])

        # Save the updated data to the JSON file
        with open(filename, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

        # Reset the mode portion of the data for the current batch
        self.data[mode] = {}


class SingleDataProcessor(DataProcessor):
    def __init__(self, mode:mode_type):
        self.mode = mode
        self.reset_data(mode)

    def reset_data(self,mode):
        self.data = {
            f"{mode}" : {}
        }
    
    def store_data(self, id: int, prediction: List[str]):
        self.data[f"{self.mode}"][id] = prediction

    def save_json(self, filename, selected_mode):
        super().save_to_json(filename=filename, mode=selected_mode)