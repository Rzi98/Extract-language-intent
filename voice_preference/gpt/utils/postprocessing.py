import json
import os
import pandas as pd
import logging
from tabulate import tabulate
import ast
import yaml

# GPT 4
PATH_NER = '../benchmark/generic_manipulation/gpt-4/ner.json'
PATH_SPLIT = '../benchmark/generic_manipulation/gpt-4/split.json'
PATH_CLASS = '../benchmark/generic_manipulation/gpt-4/classify.json'
PATH_ALL = [PATH_NER, PATH_SPLIT, PATH_CLASS]

LOG_FILEPATH = "../benchmark/generic_manipulation/gpt-4/accuracy.log"
csv_file_dir = '../benchmark/generic_manipulation/gpt-4/'
file_paths = [
    '../benchmark/generic_manipulation/gpt-4/ner.csv',
    '../benchmark/generic_manipulation/gpt-4/split.csv',
'../benchmark/generic_manipulation/gpt-4/classify.csv',
]
VISUALISATION_PATH = '../benchmark/generic_manipulation/gpt-4/result.md'


table_labels = [
    'Table 1: NER',
    'Table 2: Split',
    'Table 3: Classification',
]

PATH_DICT = {
    1: PATH_ALL,
    2: 'QUIT'
}

def postprocessing(PATH):
    '''
    Postprocessing for the benchmark results.

    SELECT THE FILE RECORD TO CHECK:
    PRINT THE RESULTS IN DICTIONARY FORMAT.
    '''
    benchmark = {} # dict stores error & total count & accuracy
    with open (PATH, 'r') as f:
        data = json.load(f)
    
    key = PATH.split('/')[-1].split('.')[0]
    misclassify = {k:v for k,v in data[key].items() if v} # return non-empty values only

    error_count = len(misclassify)
    total_data = len(data[key])
    accuracy = (total_data - error_count)/total_data
    benchmark['error_count'] = error_count
    benchmark['total_data'] = total_data
    benchmark['accuracy'] = accuracy * 100

    return misclassify, benchmark, key
        

def convert_df(hash,filename,path=csv_file_dir,transpose=True) -> None:
    '''
    Convert the dictionary into a dataframe.
    '''
    df = pd.DataFrame.from_dict(hash, orient='index')
    if transpose:
        df = df.transpose()

    if not os.path.exists(path):
        os.mkdir(path)

    df.to_csv(os.path.join(path,filename), index=True)


def replace_empty_list(lst):
    """ Function for df.apply() to replace empty list with 'NA'"""
    lst = ast.literal_eval(lst)
    if not lst:
        return 'NA'
    else:
        return lst 

# Function to read csv and convert it to markdown table
def convert_csv_to_markdown(file_path):
    """Convert csv to markdown table"""
    df = pd.read_csv(file_path)
    
    if not df.empty:
        df['predicted'] = df['predicted'].apply(replace_empty_list)

        df = df.rename(columns={'Unnamed: 0': 'id'})
        df.index += 1
        return tabulate(df, headers='keys', tablefmt='pipe')
    return 0

def convert_df_to_markdown(benchmark):
    """Convert dataframe to markdown table"""
    df = pd.DataFrame.from_dict(benchmark, orient='index', columns=['Score (%)'])
    # print(df)
    # df = df.transpose()
    return tabulate(df, headers='keys', tablefmt='pipe')

def visualise_md(file_paths, table_labels, benchmarks):
    with open(VISUALISATION_PATH, 'w') as md_file:
        for _, (file_path, label, score) in enumerate(zip(file_paths, table_labels, benchmarks)):
            # Write the table label
            md_file.write(f'<h2> {label} </h2>\n\n')
            
            # Convert CSV to Markdown and write it
            markdown_table = convert_csv_to_markdown(file_path)

            if markdown_table != 0:
                md_file.write(markdown_table)
                md_file.write('\n\n')
            markdown_score = convert_df_to_markdown(score)
            md_file.write(markdown_score)

            md_file.write(f'<h2> {label} </h2>\n\n')

if __name__ == '__main__':
    with open('./config/config.yaml', 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)


    while True:
        try:
            print("SELECT THE FILE RECORD TO CHECK:")
            print("1. ANALYSE")
            print("2. QUIT")
            print('='*150)
            choice = int(input("Enter your choice: "))
            if choice == 2:
                os.system('clear')
                print('PROGRAM TERMINATED.')
                exit()
            
            elif choice not in PATH_DICT:
                os.system('clear')
                raise ValueError("Invalid input. Please enter a number between 1 and 2.")

            else:
                PATH = PATH_DICT[choice]
                break
        except ValueError as e:
            print(e)
        except KeyError as e:
            print(e)
    
    os.system('clear')

    benchmarks = []

    for path in PATH_ALL:
        res, benchmark, task = postprocessing(path)
        benchmarks.append(benchmark)

        csv_filename = task + '.csv'
        if os.path.exists(os.path.join(csv_file_dir, csv_filename)):
            os.remove(os.path.join(csv_file_dir, csv_filename))
        convert_df(hash=res,filename=csv_filename,transpose=False)

        # ONLY LOG ACCURACY FOR CLASSIFICATION TASK
        if not os.path.exists(LOG_FILEPATH) and task.lower() == 'classify':
            logging.basicConfig(level=logging.INFO, filename=LOG_FILEPATH,format='%(asctime)s %(levelname)s %(message)s', force=True)
            logging.info(f"PATH: {PATH}")
            for k,v in benchmark.items():
                logging.info(f"{k:<11} : {v:^}")


    visualise_md(file_paths, table_labels, benchmarks)