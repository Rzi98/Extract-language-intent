import asyncio
from typing import List
from gpt.module.datatypes import (ObjectChecker, SplitChecker,
                    ClassifyChecker, InferenceChecker,
                    Correction)

async def check_obj_ext(src_obj_feat: List[str], pred_obj_feat: List[str]) -> Correction:
    obj_correction: Correction = {}
    src_obj_feat = [obj.lower() for obj in src_obj_feat]
    src_obj_feat.sort()
    pred_obj_feat = [obj.lower() for obj in pred_obj_feat]
    pred_obj_feat.sort()
    if src_obj_feat != pred_obj_feat:
        obj_correction['predicted'] = pred_obj_feat
        obj_correction['truth'] = src_obj_feat
    return obj_correction

async def check_split(src_split_res: List[str], pred_split_res: List[str])-> Correction:
    split_correction: Correction = {}
    if len(src_split_res) != len(pred_split_res):
        split_correction['predicted'] = pred_split_res
        split_correction['truth'] = src_split_res
    
    else:
        src_split_res = [obj.lower() for obj in src_split_res]
        pred_split_res = [obj.lower() for obj in pred_split_res]
        for i in range(len(src_split_res)):
            if src_split_res[i] != pred_split_res[i]:
                split_correction['predicted'] = pred_split_res
                split_correction['truth'] = src_split_res
                break
    return split_correction

async def check_classify(src_classify_res: List[str], pred_classify_res: List[str])-> Correction:
    classify_correction: Correction = {}
    src_classify_res = [obj.lower() for obj in src_classify_res]
    src_classify_res.sort()
    pred_classify_res = [obj.lower() for obj in pred_classify_res]
    pred_classify_res.sort()
    if src_classify_res != pred_classify_res:
        classify_correction['predicted'] = pred_classify_res
        classify_correction['truth'] = src_classify_res
    return classify_correction


async def inference_checker(src_data: dict, predicted_data: dict):
    """
    This function checks if the inference is correct.
    """
    data = InferenceChecker(
        object_check=ObjectChecker(source=src_data['object_list'], predicted=predicted_data['inference_objects'], correction={}), 
        split_check=SplitChecker(source=src_data['split'], predicted=predicted_data['inference_splits'], correction={}), 
        classify_check=ClassifyChecker(source=src_data['selected_features'], predicted=predicted_data['inference_class'], correction={})
        )
    
    tasks = [
        asyncio.create_task(check_obj_ext(src_obj_feat=data.object_check.source, pred_obj_feat=data.object_check.predicted)),
        asyncio.create_task(check_split(src_split_res=data.split_check.source, pred_split_res=data.split_check.predicted)),
        asyncio.create_task(check_classify(src_classify_res=data.classify_check.source, pred_classify_res=data.classify_check.predicted))
    ]

    corr_feat, corr_split, corr_class = await asyncio.gather(*tasks)

    print(f"{__name__} (corr_feat) : {corr_feat}")
    print(f"{__name__} (corr_split) : {corr_split}")
    print(f"{__name__} (corr_class) : {corr_class}")
    print()
    
    return corr_feat, corr_split, corr_class
