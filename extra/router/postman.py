from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
from dotenv import load_dotenv
import logging
import requests
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module.prompts import OBJECT_PROMPT, INTENSITY_TYPE_PROMPT


load_dotenv()

TOKEN = os.getenv('OPENAI_API_KEY')
URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-3.5-turbo"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

class UserInput(BaseModel):
    text: str
    obj_list: Optional[list[str]] = None

@app.get("/")
def index():
    logger.info(f"Post request sent to '/' url")
    return "<h1> This is the base page </h1>"

@app.post("/obj")
def object_sim(user_input: UserInput):
    logger.info(f"Post request sent to '/obj' url")

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
        }
    
    data = {
        "model": MODEL,
        "messages": [
                {
                    "role": "system",
                    "content": OBJECT_PROMPT
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
                            headers=headers,
                            data=payload)

    return json.loads(response.json()["choices"][0]["message"]["content"])


@app.post("/intensity")
def object_sim(user_input: UserInput):
    logger.info(f"Post request sent to '/obj' url")

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
        }
    
    data = {
        "model": MODEL,
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
                            headers=headers,
                            data=payload)

    return json.loads(response.json()['choices'][0]["message"]["content"])

