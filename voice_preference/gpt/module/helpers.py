import time
import os
from typing import Callable

def timer(func:Callable):
    async def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = await func(*args, **kwargs)  
        end_time = time.time()  
        execution_time = end_time - start_time  
        return result, round(execution_time,2)
    return wrapper

def timing(func:Callable):
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time()  
        execution_time = end_time - start_time  
        return result, round(execution_time,2)
    return wrapper

def clear_res_files(folder_path):
    print(folder_path)
    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Delete file
                os.remove(file_path)
                print(f"File deleted: {file_path}")

        print("Cleared all files in the folder and its subfolders.")
    except FileNotFoundError:
        print("Folder not found. Creating folder...")
        os.makedirs(folder_path)

