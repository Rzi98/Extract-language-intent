from typing import List, Optional, Literal, Callable, Union
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import re
import time
import openai
import ast

ModelName = Literal["gpt-3.5-turbo", "gpt-4"]

def timing_decorator(func:Callable):
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time()  
        execution_time = end_time - start_time  
        return result, round(execution_time,5)
    return wrapper

class GPT():

    def __init__(self, usr_cmd:str, model: Optional[ModelName]='gpt-3.5-turbo'):
        
        # (from the methods), timer
        self.res, self.ner_t = self.ner(model, usr_cmd)  # return wrapper(*args, **kwargs)
        self.object_extracted, self.obj_feat_generated, self.token_count1= self.res[0], self.res[1], self.res[2]

        with get_openai_callback() as cb:
            self.split_cmd, self.split_t = self.splitter(model, usr_cmd)
            self.token_count2 = cb.total_tokens
            
        self.predictions_and_tokens, self.classify_t = self.turbo_classifier(model, self.split_cmd, self.obj_feat_generated)
        self.predictions = self.predictions_and_tokens[0]
        self.token_count3 = self.predictions_and_tokens[1]

        (self.intensity, self.token_count4), self.intensity_t = self.get_intensity(model, self.split_cmd)

        self.hash_t = {"ner_exec_time":self.ner_t, 
                  "split_exec_time":self.split_t, 
                  "classify_exec_time":self.classify_t,
                  "intensity_exec_time":self.intensity_t,
                  "sum_exec_time":round(self.ner_t + self.split_t + self.classify_t + self.intensity_t, 5)}
    
        
    @timing_decorator
    def ner(self, model:ModelName, usr_input:str) -> Union[list,list,int]:
        
        if model == 'gpt-3.5-turbo':
            messages= [
                            {
                            "role": "system",
                            "content": "TASK:\nReturn physical/tangible object found in the input. Return 'None' if no object found.\nSeparate each object with a comma."
                            },
                            {
                            "role": "user",
                            "content": f"{usr_input} \n<list>"
                            }
                      ]
                    
        else: # GPT-4
            messages = [
                            {
                            "role": "system",
                            "content": "TASK:\nReturn physical/tangible object detected in the input. Return 'None' if no object found.\nSeparate each object with a comma."
                            },
                            {
                            "role": "user",
                            "content": f"INPUT: {usr_input} \nOBJECTS: <list>"
                            }
                        ]

        response = openai.ChatCompletion.create(
        model= model,
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        
        res = response.choices[0].message.content
        token_count = response.usage['total_tokens']

        if res is None:
            return "No object detected",[], token_count
        
        if res == "None":
            return "None",[], token_count

        formatted_text = res.replace('\n', ' ')
        split_list = formatted_text.split(', ')
        new_list = [s.replace('.', '') for s in split_list]
        dist_list = ['distance increase', 'distance decrease']
        return new_list, [f"{item1} {item2}" for item1 in new_list for item2 in dist_list], token_count
    
    @timing_decorator
    def splitter(self, model:ModelName, input_string: str) -> List[str]:
        """ Returns:
            - {str} split commands
        """

        preprocessed_str = re.sub(r'\band\b', 'AND', input_string) # Capitalise AND for better results
        preprocessed_str = re.sub(r'\bwhile\b', 'WHILE', preprocessed_str)
        preprocessed_str = re.sub(r'\bthen\b', 'THEN', preprocessed_str)

        human_message = """
        {user_command} 
        """
        if model == 'gpt-3.5-turbo':
            task_template = """ 
            Break the user sentences down into simpler sentences. 
            Separate each sentence with a comma without any newline.
            """
            prompt = ChatPromptTemplate(
                    messages=[
                        HumanMessagePromptTemplate.from_template(human_message),
                        SystemMessagePromptTemplate.from_template(task_template)
                    ],
                    input_variables=["user_command"],
                )
            chat_model = ChatOpenAI(model=model, temperature=0, max_tokens=512)
            _input = prompt.format_prompt(user_command=preprocessed_str)
            output = chat_model(_input.to_messages())
            res = output.content
            formatted_text = res.replace('\n', ' ').strip('.')
            return formatted_text.split(', ')  
        
        if model == 'gpt-4':
            response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                "role": "system",
                "content": "Break the user sentences down into simpler sentences for sentences with multiple instructions. Otherwise return the same input. \nSeparate each sentence with a comma without any newline."
                },
                {
                "role": "user",
                "content":  f"INPUT: {preprocessed_str} \nOUTPUT: <list>"
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )    

            res = response.choices[0].message.content
            formatted_text = res.replace('\n', ' ').strip('.')
            return formatted_text.split(', ')
        
    @timing_decorator
    def turbo_classifier(self, model:ModelName, processed_commands:str, dynamic_features:list) -> Union[list,int]:
        features = ["Z-cartesian increase", "Z-cartesian decrease", 
                            "Y-cartesian increase", "Y-cartesian decrease", 
                            "X-cartesian increase", "X-cartesian decrease"] + dynamic_features

        if model == 'gpt-3.5-turbo':
            messages = [
                            {
                            "role": "system",
                            "content": "TASK:\n    Return the closest semantic feature from Features that matches each command in a list. \n    If approach an object, object distance should decrease. \n    If avoid an object, object distance should increase.\n    Output it into a list separated by comma.\nLength of output == length of input.\nOutput must be from Features.\n\n   REFERENCE AXES: \n    Key: left , Feature: X-cartesian decrease\n    Key: right , Feature: X-cartesian increase\n\n    Key: up , Feature: Z-cartesian increase\n    Key: down , Feature: Z-cartesian decrease \n\n    Key: forward , Feature: Y-cartesian increase\n    Key: backward , Feature: Y-cartesian decrease"
                            },
                            {
                            "role": "user",
                            "content": f"INPUT:  \n{processed_commands}\n\FEATURES:\n{features}\n\OUTPUT:\n<list>"
                            },
                        ]
    
        else: # GPT-4
            messages = [
                            {
                            "role": "system",
                            "content": "TASK:\n    Return the closest semantic feature from Features that matches each command in a list. \n    If approach an object, object distance should decrease. \n    If avoid an object, object distance should increase.\n    Output it into a list separated by comma.\nLength of output == length of input.\nOutput must be from Features.\n\n   REFERENCE AXES (when commands does not involve object): \n    Key: left , Feature: X-cartesian decrease\n    Key: right , Feature: X-cartesian increase\n\n    Key: up , Feature: Z-cartesian increase\n    Key: down , Feature: Z-cartesian decrease \n\n    Key: forward , Feature: Y-cartesian increase\n    Key: backward , Feature: Y-cartesian decrease"
                            },
                            {
                            "role": "user",
                            "content": f"INPUT:  \n{processed_commands}\n\FEATURES:\n{features}\n\OUTPUT:\n<list>"
                            },
                        ]
            
        response = openai.ChatCompletion.create(
        model= model,
        messages=messages,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

        token_count = response.usage['total_tokens']
        
        res = response.choices[0].message.content
        res = res.replace('\n', ' ').strip('.')

        if isinstance(res, str):
            res = ast.literal_eval(res)
        if ',' in res:
            res = res.split(',')
        return res, token_count
    
    @timing_decorator
    def get_intensity(self, model:ModelName, processed_commands:list) -> Union[list,int]:

        messages = [
                    {
                    "role": "system",
                    "content":  "Rank each of the input commands based on the adverb modifiers which specify the intensity of distance that is required to perform the action.\n\nOptions = ['low', 'neutral', 'high']\n\nStrong modifier:\n'Extremely' : high\n\nLight modifier:\n'slightly' : low\n\nNo modifier : neutral\n\nOnly output the answer in a list: list(str(output))"
                    },
                    {
                    "role": "assistant",
                    "content": f"{processed_commands}"
                    }
                    ]

        response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        res = response.choices[0].message.content
        res = ast.literal_eval(res)

        output = {}
        for a, b in zip(processed_commands, res):
            output[a] = b

        token_count = response.usage['total_tokens']

        return output, token_count
        
    
    def result_parser(self):
        return (self.object_extracted,self.obj_feat_generated), self.split_cmd, self.predictions, self.intensity, (self.token_count1, self.token_count2, self.token_count3, self.token_count4), self.hash_t
    