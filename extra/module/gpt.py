import os
import json
import sys
import asyncio
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import (Optional, Literal, ClassVar, Union, TypedDict,
                    Dict, Any)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module.prompts import OBJECT_PROMPT, INTENSITY_TYPE_PROMPT
from module.datatypes import (Data, UserMessage, model)

load_dotenv()

TOKEN = os.getenv('OPENAI_API_KEY')
URL = "https://api.openai.com/v1/chat/completions"
CONTENT_TYPE = "application/json"

class Model:
    headers: ClassVar[dict] = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": CONTENT_TYPE
        }

    def __init__(self, model: Optional[model]='gpt-3.5-turbo'):
        self.model = model
    
    def get_closet_obj(self, user_input: UserMessage) -> Dict[str, Any]:
        data: Dict[str, Any] = {
        "model": self.model,
        "messages": [
                {
                    "role": "system",
                    "content": f"{OBJECT_PROMPT}"
                },
                {
                    "role": "user",
                    "content": f"{user_input.text} \nobj_list={user_input.obj_list}"
                },
                
            ],
            "temperature": 0,
            "max_tokens": 512,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        payload = json.dumps(data)
        
        response = requests.request(method='POST',
                                url=URL,
                                headers=Model.headers,
                                data=payload)
        if response.status_code == 200:
            return json.loads(response.json()["choices"][0]["message"]["content"])
        
        else:
            raise Exception(f"Error: {response.status_code}")

    def get_intensity_type(self, user_input: UserMessage) -> Dict[str, Any]:
        data: Dict[str, Any] = {
        "model": self.model,
        "messages": [
                {
                    "role": "system",
                    "content": INTENSITY_TYPE_PROMPT
                },
                {
                    "role": "user",
                    "content": user_input.text
                },
                
            ],
            "temperature": 0,
            "max_tokens": 512,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        payload = json.dumps(data)
        response = requests.request(method='POST',
                                url=URL,
                                headers=Model.headers,
                                data=payload)

        if response.status_code == 200:
            return json.loads(response.json()["choices"][0]["message"]["content"])
        
        else:
            raise Exception(f"Error: {response.status_code}")
    
    @staticmethod
    def data_to_user_input(data: Data) -> Union[UserMessage, int]:
        test_id:int = data.id
        return UserMessage(text=data.corrections, obj_list=data.obj_names), test_id


if __name__ == "__main__":

    os.system('clear')
    test_data = Data(id=1, corrections='stay closer to the Egyptian cat',
                    obj_names=['acoustic guitar', 'RV', 'trolley', 'minibus', 'Egyptian cat', 'European fire salamander'], change_type='dist')
    
    test_msg, test_id =Model.data_to_user_input(test_data)

    m = Model(model='gpt-3.5-turbo')
    print(m.get_closet_obj(test_msg))