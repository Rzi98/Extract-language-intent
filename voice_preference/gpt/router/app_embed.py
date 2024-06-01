from flask import Flask, render_template  
from pathlib import Path
from typing import Optional
import pandas as pd  
import json
  
app = Flask(__name__)  

def load_json(file:Optional[Path]='../result/embeddings/ada.json'):
    with open(file=file, mode='r') as f:
        data = json.load(f)
    return data

def load_dataframe(data:dict):
    corrections = {}
    for id in data:
        correction = data[str(id)].pop("correction") 
        corrections[id] = correction
    df_corr = pd.DataFrame.from_dict(corrections, orient='columns').T
    df_corr.reset_index(inplace=True) 
    rows = []  
    for id, values in data.items():  
        segment = 1  
        for cmd, cmd_values in values.items():
            if isinstance(cmd_values, dict):  
                row = {"id": id,"segment": segment, "cmd": cmd}  
                row.update(cmd_values)  
                rows.append(row)  
                segment += 1 

    df = pd.DataFrame(rows)
    new_df = pd.merge(df, df_corr, left_on='id', right_on='index', how='left')
    new_df.drop(columns=['index'], inplace=True)

    new_df['predicted'] = new_df['predicted'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)  
    new_df['truth'] = new_df['truth'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)  
    
    new_df = new_df.groupby('id', as_index=False).agg({  
        'segment': list,  
        'cmd': list,  
        'closest_pattern': list,  
        'predicted_feature': list,  
        'confidence_score': list,  
        'predicted': 'first',  
        'truth': 'first'  
    })
    new_df['id'] = pd.to_numeric(new_df['id'])
    new_df=new_df.sort_values('id', inplace=False) 
    new_df.fillna('-', inplace=True)  
    new_df.drop(columns=['segment'], inplace=True)
 
    return new_df

@app.route("/")  
def home():  
    data = load_json()
    new_df = load_dataframe(data) 
    table = new_df.to_html(justify='center', classes='table table-striped', index=False)  
    return render_template('embed.html', table=table)  
  
if __name__ == '__main__':  
    while True:
        try:
            app.run(debug=True, host='127.0.0.1', port=5001)  
        
        except SyntaxError as e:
            print(e)
            continue
