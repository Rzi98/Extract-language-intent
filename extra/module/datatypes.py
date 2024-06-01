from pydantic import BaseModel
from typing import Literal, TypedDict, Optional, Any, List
import itertools
import json
import os

model = Literal["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]
intensity = Literal["low", "neutral", "high"]
change_type = Literal["distance", "speed", None]
direction = Literal["increase", "decrease", None]
object_name = str | None
cartesian_axes = str | None
classification = str | None

class Data(BaseModel):
    id: int
    corrections: str
    obj_list: list[str]
    gt_type: list[change_type]
    gt_target_obj: list[object_name]
    gt_direction: list[direction]
    gt_intensity: list[intensity]
    gt_cart_axes: list[cartesian_axes]
    gt_split_commands: list[str]
    gt_classification: list[classification]


class UserMessage(BaseModel):
    text: str
    obj_list: Optional[list[object_name]] = None

class SimilarityObjResponse(TypedDict):
    similarity: dict[object_name, float]
    closest_obj: object_name

class IntensityTypeResponse(TypedDict):
    intensity: intensity
    type: change_type

## DATA ## 
class MyData(BaseModel):
    commands: str
    id: int
    split_commands: list[str]
    dynamic_features: list[str | None]

class Result(BaseModel):
    id: int
    corrections: str
    obj_names: list[object_name]
    change_type: change_type
    similarity: dict[object_name, float]
    target_obj: object_name
    direction: direction
    intensity: intensity
    cart_axes: cartesian_axes
    predicted_feature: classification


if __name__ == '__main__':
    os.system('clear')

    with open("../data/100k_sample.json") as f:
        data = json.load(f)
    
    start_index = 1
    end_index = 5

    sliced_data = dict(itertools.islice(data.items(), start_index-1, end_index))
    test_cases = [
        Data(id=int(k)+1,
             corrections=v['text'], 
             obj_names=v['obj_names'],
             change_type=v['change_type']) 
        for k,v in sliced_data.items()]
    
    print(test_cases)
    
    