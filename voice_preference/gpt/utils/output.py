### Create a json file for the output of GPT4 on Flask ###

import yaml
import json


def get_output():

    with open('../config/config.yaml', 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    path_data = config['DIRECTORY']['VISUALISATION']['GPT4']['INPUT']['DATA']
    path_features = config['DIRECTORY']['VISUALISATION']['GPT4']['INPUT']['PATH_FEATURE']
    path_intensity = config['DIRECTORY']['VISUALISATION']['GPT4']['INPUT']['PATH_INTENSITY']
    output_path = config['DIRECTORY']['VISUALISATION']['GPT4']['OUTPUT']['JSON_PATH']

    with open(path_data, 'r') as json_file:
        data = json.load(json_file)

    with open(path_features, 'r') as json_file:
        feat_lookup = json.load(json_file)['predicted_features']

    with open(path_intensity, 'r') as json_file:
        intense_lookup = json.load(json_file)['predicted_intensity']

    output_dict = {}

    for idx, key in enumerate(data.keys()):
        output_dict[idx+1] = {}
        output_dict[idx+1]['input'] = key
        output_dict[idx+1]['features'] = feat_lookup[str(idx+1)]
        output_dict[idx+1]['intensity'] = list(intense_lookup[str(idx+1)].values())


    for k,v in output_dict.items():
        print(f"{k} : {v}")

    with open(output_path, 'w') as json_file:
        feat_lookup = json.dump(output_dict,json_file,indent=4)

if __name__ == "__main__":
    get_output()
