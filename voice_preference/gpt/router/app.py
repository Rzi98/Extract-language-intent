from flask import Flask, render_template
import json
import yaml

app = Flask(__name__)

def get_path():
    with open('../config/config.yaml', 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    path_data = config['VISUALISATION']['RESULT']
    return path_data

@app.route('/')
def index():
    path_data = get_path()
    with open(path_data, 'r') as json_file:
        data = json.load(json_file)
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)