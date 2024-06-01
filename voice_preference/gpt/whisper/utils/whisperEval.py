import json
import os
import string
import pandas
from pathlib import Path
from typing import Optional
import pandas as pd

def retrieveData(candidate:str, res_types:str) -> json:
    GT_PATH: Path = Path(os.path.join(os.path.dirname(CWD), "dataset/gt.json"))
    RES_PATH: Path = Path(os.path.join(os.path.dirname(CWD), f"results/{res_types}/{candidate}.json"))
    
    with open(file=GT_PATH, mode='r') as f:
        gt = json.load(f)
    with open(file=RES_PATH, mode='r') as f:
        res = json.load(f)

    for i in range(len(res)):
        res[i]['rectified_input'] = res[i]['rectified_input'].strip(string.punctuation).replace(' the', '')
        for j in range(len(res[i]['intensity_list'])):
            res[i]['intensity_list'][j].pop(0)
        for j in range(len(res[i]['split_command'])):
            res[i]['split_command'][j] = res[i]['split_command'][j].replace(' the', '')
    
    for i in range(len(gt)):
        gt[i]['transcription_gt'] = gt[i]['transcription_gt'].strip(string.punctuation).replace(' the', '')
        for j in range(len(gt[i]['intensity_list_gt'])):
            gt[i]['intensity_list_gt'][j].pop(0)
        for j in range(len(gt[i]['split_commands_gt'])):
            gt[i]['split_commands_gt'][j] = gt[i]['split_commands_gt'][j].replace(' the', '')
            
    return gt, res

def compare(gt:list, res:list, candidate: str, verbose=False, metric_type=None) -> None:
    res_keys = ["rectified_input", "object_list", "split_command", "classification", "intensity_list"]
    gt_keys = ["transcription_gt", "objects_gt", "split_commands_gt", "classification_gt", "intensity_list_gt"]

    hashmap = {i+1 : {} for i in range(len(gt))}

    for i in range(len(gt)):
        if verbose:
            print(f"ID: {gt[i]['id']}")
        for res_key, gt_key in zip(res_keys, gt_keys):
            if gt[i][gt_key] == res[i][res_key]:
                continue
            else:
                hashmap[i+1][res_key] = res[i][res_key] 
                hashmap[i+1][gt_key] = gt[i][gt_key]
                if verbose:
                    print(f"ERROR FIELD: {res_key}")
                    print(f"RESULT: {res[i][res_key]}")
                    print(f"ACTUAL: {gt[i][gt_key]}", end='\n\n')
        if verbose:
            print('=' * 150, end='\n\n')

    if metric_type is not None:
        if not os.path.exists(f'../metrics/{metric_type}'):
            os.makedirs(f'../metrics/{metric_type}')
        output_path = f"../metrics/{metric_type}/{candidate}_eval.json"
        with open(file=output_path, mode='w') as f:
            json.dump(hashmap, f, indent=4)
    else:
        raise ValueError("Metric type not found")
    
def get_metrics(candidate:str, metric:str) -> dict:
    df= pd.read_json(f'../metrics/{metric}/{candidate}_eval.json').T
    success = df.index[df['classification'].isna()].tolist()
    feature_acc = round((len(success)/len(df)) * 100, 2)

    failure = df.index[~df['classification'].isna()].tolist()

    sub_df = df.loc[failure]

    failure_obj = sub_df.index[~sub_df['object_list'].isna()].tolist()

    transc_error_idx = df.index[~df['rectified_input'].isna()].tolist()
    correct_idx = df.index[df['rectified_input'].isna()].tolist()

    correct_transcription_count = len(correct_idx)
    incorrect_transcription_count = len(transc_error_idx)
    
    for i in reversed(transc_error_idx):
        rectified_input = df.loc[i]['rectified_input'].translate(str.maketrans('', '', string.punctuation)).lower().strip()
        ground_truth = df.loc[i]['transcription_gt'].translate(str.maketrans('', '', string.punctuation)).lower()
        if rectified_input != ground_truth:
            
            incorrect_transcription_count += 1
            print(transc_error_idx)
            
        else:
            correct_transcription_count += 1
            transc_error_idx.remove(i)

    transcript_acc = round((correct_transcription_count/len(df)) * 100, 2)

    print(f"TRANSCRIPTION ACCURACY: {transcript_acc}%")
    print(f"FEATURE ACCURACY: {feature_acc}%")

    res = {
        "id" : candidate,
        "success_idx" : success,
        "failure_obj_idx" : failure_obj,
        "failure_transc_idx" : transc_error_idx,
        "transcript_accuracy" : f"{transcript_acc}%",
        "feature_accuracy" : f"{feature_acc}%"
    }
    return res

def run(candidate_name: str, metric: str) -> None:
    gt, res = retrieveData(candidate=candidate_name, res_types=metric)
    compare(gt, res, candidate=candidate_name, verbose=False, metric_type=metric)

def run_metrics(metric:None) -> None:
    res = []
    for name in os.listdir("../confidential_data"):
        indiv_res = get_metrics(candidate=name, metric=metric)
        res.append(indiv_res)
    
    if not os.path.exists(f'../metrics/score'):
        os.makedirs(f'../metrics/score')

    with open(file=f'../metrics/score/{metric}.json', mode='w') as f:
        json.dump(res, f, indent=4)

if __name__ == "__main__":

    # metric = "semantic"
    metric = "sound"

    CWD: Path = Path(os.path.dirname(__file__))
    for candidate in os.listdir(os.path.join(os.path.dirname(CWD), "confidential_data")):
        # print(candidate)

        run(candidate_name=candidate, metric=metric)

    run_metrics(metric=metric)


