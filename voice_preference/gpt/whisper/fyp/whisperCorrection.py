import os
import json
import requests
import jellyfish
from dotenv import load_dotenv
# from apiModel import APIWhisper
from typing import Optional
import aiohttp 

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
    def rectify_transcription(cls, transcription: str, vision_list: Optional[list[str]]= [], prompt_type=None) -> str:

        transcription = transcription.strip()

        sys_prompt1 = f"""
                        Replace the noun object with item from gt_list with the closest semantic meaning as `rectified_sentence`. 
                        If no object is matched, return the original sentence as `rectified_sentence`.
                        If the noun object is in gt_list, return the original sentence as `rectified_sentence`.

                        rectified_sentence must be a command. 

                        ground_truth_object = {vision_list}


                        JSON FORMAT:
                        {{
                            "wrong_obj" : "____",
                            "correct_obj" : "____",
                            "rectified_sentence" : "____",
                        }}
                    """
        sys_prompt2 = f"""
                        Rectify and return the user message by replacing the noun object with the item from gt_list with the closest sound as `rectified_sentence`. 
                        If no object is matched, return the original sentence as `rectified_sentence`.

                        rectified_sentence must be a command.

                        ground_truth_object = {vision_list}

                        JSON FORMAT:
                        {{
                        "wrong_obj" : "____",
                        "correct_obj" : "____",
                        "rectified_sentence" : "____",
                        }}
                    """
        
        if prompt_type == "semantic":
            sys_prompt = sys_prompt1
        elif prompt_type == "sound":
            sys_prompt = sys_prompt2
        else:
            raise ValueError("Invalid prompt type. Please specify either 'semantic' or 'sound'.")

        print(f"Prompt Type Correction: {prompt_type}")
        body = {
                    "model": "gpt-4-1106-preview",
                    "messages": [
                            {
                                "role": "system",
                                "content": sys_prompt
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
            corrected_txt = json.loads(response['choices'][0]['message']['content'])

            return list(corrected_txt.values())[-1]
        

def improve_with_gpt():

    print("GPT4")
    gt_list = ["scissors", "mug", "vase", "bottle", "bowl", "orange", "laptop", "banana", "flower"]
    print(gt_list)

    transcribed_txt = "Move a bit closer to the bow, but stay away from the flask."
    print(transcribed_txt)
    output = WhisperCorrection.rectify_transcription(transcription=transcribed_txt, vision_list=gt_list, prompt_type="semantic")

    print(output)

    

def improve_with_jellyfish():
    llm_objs = ["bus", "orange"]
    obj_vision = ["mug", "orange", "carrot", "vase", "elephant"]

    print("Jaro Similarity")
    # print(llm_objs)
    # print(obj_vision)

    # for obj in llm_objs:
    #     if obj not in obj_vision:
    #         a = [jellyfish.jaro_similarity(obj, obj_gt) for obj_gt in obj_vision]
    #         print(f"Object: {obj}\nSimilarity: {a}\nClosest Match: {obj_vision[a.index(max(a))]}")
    #         b = [jellyfish.jaro_winkler_similarity(obj, obj_gt) for obj_gt in obj_vision]
    #         print(f"Object: {obj}\nSimilarity: {b}\nClosest Match: {obj_vision[b.index(max(b))]}")

    a = jellyfish.soundex("notebook")
    b = jellyfish.soundex("goodbook")

    print(jellyfish.jaro_winkler_similarity(a, b))

if __name__ == '__main__':
    # print("="*100)
    improve_with_gpt()
    # print("="*100)
    # improve_with_jellyfish()