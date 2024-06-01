import json
import os
from pathlib import Path
from typing import Optional

def load_data(path: Optional[Path]="../data/100k_sample.json") -> dict:
    with open(file=path, mode='r') as f:
        data = json.load(f)
    return data

def save_data(data: dict, path: Optional[Path]="../data/data.json") -> None:
    with open(file=path, mode="w") as f:
        json.dump(data, f, indent=4)

def generate_new_dict(data: dict) -> dict:
    new = {}
    for id in data:
        inner = {}
        inner['correction'] = data[id]['text']
        inner['obj_list'] = data[id]['obj_names']

        inner['gt_type'] = data[id]['gt_change_type']


        if data[id]['gt_target_object'] == [""]:
            inner['target_obj'] = [None]
        else:
            inner['gt_target_object'] = data[id]['gt_target_object']

        direction_list = [direction.lower() for direction in data[id]['gt_direction']]
        inner['gt_direction'] = direction_list

        if data[id]['gt_intensity'] == ["-"]:
            inner['gt_intensity'] = [None]
        else:
            intensity_list = [intensity.lower() for intensity in data[id]['gt_intensity']]
            inner['gt_intensity'] = intensity_list

        if data[id]['gt_cart_axes'] == ["-"]:
            inner['gt_cart_axes'] = [None]
        else:
            inner['gt_cart_axes'] = data[id]['gt_cart_axes']
            
        inner['gt_split'] = data[id]['gt_split']
        inner['gt_classification'] = data[id]['gt_feature']
        new[str(int(id)+1)] = inner
    return new

def refine_data():
    data = load_data()
    new_data = generate_new_dict(data)
    save_data(new_data)
    print("Done")

def update_data(main_path: Optional[Path]="../data/test_set.json",
                path_to_add: Optional[Path]="../data/updates.json") -> None:
    data = load_data(path=main_path)
    with open(file=path_to_add, mode='r') as f:
        new_data = json.load(f)
    
    data.update(new_data)
    save_data(data=data, path=main_path)
    print("Done")

if __name__ == '__main__':
    os.system('clear')
    # refine_data()
    update_data()
 
    
    