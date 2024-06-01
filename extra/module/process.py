from pydantic import BaseModel
import os, json, requests, itertools, sys
from typing import Optional, ClassVar, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.prompts import (CLASSIFIER_PROMPT, CLASSIFIER_FORMAT,
                                     CARTESIAN_REFERENCE_TEMPLATE)
from module.datatypes import (Data, Result, SimilarityObjResponse,
                                       IntensityTypeResponse, direction)
from module.gpt import (Model, UserMessage, TOKEN, URL,
                                 CONTENT_TYPE, model) 



CARTESIANS_FEATURES = [
    "Z-cartesian increase", "Z-cartesian decrease", 
    "Y-cartesian increase", "Y-cartesian decrease", 
    "X-cartesian increase", "X-cartesian decrease"
]

class PredictData(BaseModel):
    """
        initial result containing:
        1) similarity scores
        2) closest object - object with highest similarity score
        3) intensity type - [low, neutral, high]
        4) change type - [dist, speed]
    """
    id: int
    similarity: dict
    closest_obj: str
    intensity: str
    type: str

class FeaturesLabels(PredictData):
    cart_feat: Optional[List[str]] = None
    speed_feat: Optional[List[str]] = None
    dist_feat: Optional[List[str]] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.type.lower() == "none" or self.closest_obj.lower() == "none":
            self.cart_feat = CARTESIANS_FEATURES
            self.speed_feat = None
            self.dist_feat = None
        elif self.type.lower() == "distance":
            self.cart_feat = None
            self.speed_feat = None
            self.dist_feat = [f"{self.closest_obj} distance increase", f"{self.closest_obj} distance decrease"]
        else:
            self.cart_feat = None
            self.speed_feat = ["speed increase","speed decrease"]
            self.dist_feat = None

class Processing:
    headers: ClassVar[dict] = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": CONTENT_TYPE
        }

    @staticmethod
    def dict_to_PredictData(id:int, sim_dict: dict, intensity_dict: dict) -> PredictData:
        return PredictData(id=id, similarity=sim_dict["similarity"], closest_obj=sim_dict["closest_obj"], intensity=intensity_dict["intensity"], type=intensity_dict["type"])
    
    @staticmethod
    def classify(user_input: UserMessage, data: FeaturesLabels,
                model: Optional[model] = "gpt-3.5-turbo") -> json:
        
        ## CARTESIAN ##
        if data.type.lower() == "none" or data.closest_obj.lower() == "none":
            body: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                            {
                                "role": "system",
                                "content": CLASSIFIER_PROMPT + CARTESIAN_REFERENCE_TEMPLATE + CLASSIFIER_FORMAT
                            },
                            {
                                "role": "user",
                                "content": user_input.text + "\n" + str(data.cart_feat)
                            },
                            
                        ],
                        "temperature": 0,
                        "max_tokens": 512,
                        "top_p": 1,
                        "frequency_penalty": 0,
                        "presence_penalty": 0
                    }
        elif data.type.lower() == 'distance':
            body = {
                    "model": model,
                    "messages": [
                            {
                                "role": "system",
                                "content": CLASSIFIER_PROMPT + CLASSIFIER_FORMAT
                            },
                            {
                                "role": "user",
                                "content": user_input.text + "\n" + str(data.dist_feat)
                            },
                            
                        ],
                        "temperature": 0,
                        "max_tokens": 512,
                        "top_p": 1,
                        "frequency_penalty": 0,
                        "presence_penalty": 0
                    }
        else:
            body = {
                    "model": model,
                    "messages": [
                            {
                                "role": "system",
                                "content": CLASSIFIER_PROMPT + CLASSIFIER_FORMAT
                            },
                            {
                                "role": "user",
                                "content": user_input.text + "\n" + str(data.speed_feat)
                            },
                            
                        ],
                        "temperature": 0,
                        "max_tokens": 512,
                        "top_p": 1,
                        "frequency_penalty": 0,
                        "presence_penalty": 0
                    }

        payload = json.dumps(body)
        response = requests.request(method='POST',
                                url=URL,
                                headers=Model.headers,
                                data=payload)
        
        if response.status_code == 200:
            print(response.json()["choices"][0]["message"]["content"])
            classify_dict: dict = json.loads(response.json()["choices"][0]["message"]["content"])
            if data.type.lower() == "none" or data.closest_obj.lower() == "none":
                dir: direction = "none"
            else:
                dir: direction = classify_dict['output'].split()[-1] # increase/decrease

            return Result(id=data.id, corrections=user_input.text, obj_names=user_input.obj_list, change_type=data.type, similarity=data.similarity, target_obj=data.closest_obj, direction=dir, intensity=data.intensity, cart_axes=data.cart_feat, predicted_feature=classify_dict["output"])

        else:
            raise Exception(f"Error: {response.status_code}")

if __name__ == '__main__':
    os.system('clear')
    
    with open("../data/100k_sample.json") as f:
        data = json.load(f)
    
    start_index = 3
    end_index = 4
    sliced_data = dict(itertools.islice(data.items(), start_index-1, end_index))
    test_cases = [
        Data(id=int(k)+1,
             corrections=v['text'], 
             obj_names=v['obj_names'],
             change_type=v['change_type']) 
        for k,v in sliced_data.items()]
    
    lookup = {}
    for test_data in test_cases:

        test_msg, test_id = Model.data_to_user_input(test_data)

        m = Model(model='gpt-3.5-turbo')

        sim_dict: SimilarityObjResponse = m.get_closet_obj(test_msg)
        intensity_dict: IntensityTypeResponse = m.get_intensity_type(test_msg)
        predicted_dict = Processing.dict_to_PredictData(id=test_id, sim_dict=sim_dict, intensity_dict=intensity_dict).model_dump()
        fl = FeaturesLabels(id=predicted_dict["id"], similarity=predicted_dict["similarity"],
                            closest_obj=predicted_dict["closest_obj"], intensity=predicted_dict["intensity"], 
                            type=predicted_dict["type"])

        result = Processing.classify(user_input=test_msg, data=fl, model='gpt-3.5-turbo').model_dump()
        index = result.pop("id")
        lookup[index] = result

    print(lookup)