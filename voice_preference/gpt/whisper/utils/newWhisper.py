import sys
import os
import json
import requests
import jellyfish
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from fyp.apiModel import APIWhisper
from typing import TypedDict

load_dotenv()


class WhisperCorrection:
    TOKEN = os.getenv('OPENAI_API_KEY')
    URL = "https://api.openai.com/v1/chat/completions"
    CONTENT_TYPE = "application/json"
    HEADERS: dict = {
                            "Authorization": f"Bearer {TOKEN}",
                            "Content-Type": CONTENT_TYPE
                        }
    
    @classmethod
    def rectify_transcription(cls, transcription: str, vision_list: list[str]) -> str:

        
        sys_prompt1 = f"""
                        Rectify the user message by replacing the object with object of the closest semantic meaning from gt_list.

                        ground_truth_object = {vision_list}

                        JSON FORMAT:
                        {{
                            "wrong_object" : "____",
                            "correct_object" : "____",
                            "rectified_sentence" : "____",
                        }}
                    """
        sys_prompt2 = f"""
                        Rectify the user message by replacing the object with the closest sound object from gt_list.

                        ground_truth_object = {vision_list}

                        JSON FORMAT:
                        {{
                            "wrong_object" : "____",
                            "correct_object" : "____",
                            "rectified_sentence" : "____",
                        }}
                    """
        
        body = {
                    "model": "gpt-4-1106-preview",
                    "messages": [
                            {
                                "role": "system",
                                "content": sys_prompt2
                            },
                            {
                                "role": "user",
                                "content": transcription
                            }      
                        ],
                        "temperature": 0,
                        "max_tokens": 512,
                        "top_p": 1,
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "response_format": { "type": "json_object" },
                        "seed": 0
                    }
        
        response = requests.request(method='POST',
                                url=cls.URL,
                                headers=cls.HEADERS,
                                data=json.dumps(body))
        
        if response.status_code == 200:
            response = response.json()
            print(response)
            return json.loads(response['choices'][0]['message']['content'])
        

def improve_with_gpt():

    ## TODO: Similar sounding 
    ## TODO: Similar meaning (synonyms)

    print("GPT4")
    gt_list = ["scissors", "mug", "vase", "bottle", "bowl", "orange", "laptop", "banana", "flower"]
    print(gt_list)

    # transcribed_txt = APIWhisper.transcribe(filename="../audio_files/Rhonda/recording_2.wav")
    transcribed_txt = "Move fast when closer to the bus"
    print(f"Raw Transcription: {transcribed_txt}")
    rectified_txt = list(WhisperCorrection.rectify_transcription(transcription=transcribed_txt, vision_list=gt_list).values())
    print(f"Rectified Transcription: {rectified_txt}")

def improve_with_jellyfish():
    llm_objs = ["bus", "orange"]
    obj_vision = ["scissors", "mug", "vase", "bottle", "bowl", "orange", "laptop", "banana", "flower"]

    print("Jaro Similarity")
    print(llm_objs)
    print(obj_vision)

    for obj in llm_objs:
        if obj not in obj_vision:
            a = [jellyfish.jaro_similarity(obj, obj_gt) for obj_gt in obj_vision]
            print(f"Object: {obj}\nSimilarity: {a}\nClosest Match: {obj_vision[a.index(max(a))]}")


if __name__ == '__main__':
    # print("="*100)
    improve_with_gpt()
    # print("="*100)
    # improve_with_jellyfish()

    # print(jellyfish.soundex("flask"))
    # print(jellyfish.soundex("vase"))