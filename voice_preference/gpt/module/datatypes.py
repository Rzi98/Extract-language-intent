from pydantic import BaseModel, Field
from typing import Literal, List, NewType, Dict
from pathlib import Path

model_version = Literal["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]
Correction = NewType('Correction', Dict[str, str]) # Dict['predicted', 'truth']

class ConfigArgument(BaseModel):
    model : model_version = Field(description="Model to be used for inference")
    batch_no : int = Field(description="Batch of data for inference", values={1, 2, 3, 4, 5, 9})
    batches : dict = Field(description="Dictionary of batch ranges")

class InputPath(BaseModel):
    data : Path = Field(description="Path to the data file")

class OutputPath(BaseModel):
    logger: Path = Field(description="Path to the log file")
    result: Path = Field(description="Path to the result folder")

class BaseChecker(BaseModel):  
    source: List[str]  
    predicted: List[str]  
    correction: dict

class ObjectChecker(BaseChecker):
    pass

class SplitChecker(BaseChecker):
    pass

class ClassifyChecker(BaseChecker):
    pass

class InferenceChecker(BaseModel):
    object_check: ObjectChecker
    split_check: SplitChecker
    classify_check: ClassifyChecker