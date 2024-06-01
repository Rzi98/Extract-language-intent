import json
import os

# INTENSITY_PATH = '../result/gpt-4-1106-preview/prediction/intensity_w_types.json'
# CLASSIFICATION_PATH = '../result/gpt-4-1106-preview/prediction/classification_w_scores.json'
# DATASET = '../data/new_generic.json'
# OUTPUT = '../result/gpt-4-1106-preview/prediction/reformatted.json'

INTENSITY_PATH = '../exp_50/gpt-4-1106-preview/prediction/intensity_w_types.json'
CLASSIFICATION_PATH = '../exp_50/gpt-4-1106-preview/prediction/classification_w_scores.json'
DATASET = '../data/new_generic.json'
OUTPUT = '../exp_50/gpt-4-1106-preview/prediction/reformatted.json'

def reformat():
    """Reformat the prediction json file to be more readable for FastAPI"""
    
    with open(INTENSITY_PATH, 'r') as f:
        dict1 = json.load(f)
    with open(CLASSIFICATION_PATH, 'r') as f:
        dict2 = json.load(f)
    
    with open(DATASET, 'r') as f:
        data = json.load(f)
    
    input_dict = {str(count): key for count, key in enumerate(data, start=1)}  


    new_dict = {}  
    for key in dict1["intensity_w_types"]:  
        input_str = input_dict[key]
        features = [item[0] for item in dict2["classification_w_scores"][key]]  
        confidence = [item[1] for item in dict2["classification_w_scores"][key]]
        intensity = [item[1].upper() for item in dict1["intensity_w_types"][key]]  
        types = [item[2].upper() for item in dict1["intensity_w_types"][key]]  
        new_dict[key] = {"input": input_str, "features": list((x,y) for x,y in zip(features, confidence)), 
                         "intensity": intensity, "types": types}
    
    with open(OUTPUT, 'w') as f:
        json.dump(new_dict, f, indent=4)

if __name__ == '__main__':
    os.system('clear')
    reformat()