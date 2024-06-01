from flask import Flask, render_template
from pathlib import Path
import os
import json
import yaml

app = Flask(__name__)

def get_path(filepath:Path='../config/config.yaml'):
    with open(file=filepath, mode='r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        directory = config['VIEW']
        return directory

def read_result(directory:Path, mode:str='r' ):
    with open(directory, mode) as json_file:
        data = json.load(json_file)
        return data

  
@app.route('/', methods=['GET'])
def index() -> dict:  
    os.system('clear')
    directory = get_path()
    data = read_result(directory)  
    return render_template('view.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)