import json
import pandas as pd

def extract_time():
    gpt_path = '../TIME/gpt-4-1106-preview/prediction/timer.json'
    gpt_50_path = '../exp_50/gpt-4-1106-preview/prediction/timer.json'
    hybrid_path = '../TIME/embeddings/ada.json'
    hybrid_50_path = '../exp_50/embeddings/ada.json'

    gpt_timers = {}
    hybrid_timers = {}
    temp = []

    with open(gpt_path, 'r') as f:
        data = json.load(f)['timer']
        gpt_100 = sum([float(x) for x in data.values()]) 
    
    for i, time in enumerate([float(x) for x in data.values()]):
        gpt_timers[i+1] = time
    
    with open(gpt_50_path, 'r') as f:
        data = json.load(f)['timer']
        gpt_50 = sum([float(x) for x in data.values()]) 
    
    for i, time in enumerate([float(x) for x in data.values()]):
        gpt_timers[i+101] = time
    
    gpt_avg_t = (gpt_100 + gpt_50) / 150 
    print(f'GPT-4 AVG inference time: {gpt_avg_t:.2f} seconds')

    with open(hybrid_path, 'r') as f:
        data = json.load(f)
        total = 0
        for _, v in data.items():
            total += v['elapsed_time']
            temp.append(v['elapsed_time'])

    with open(hybrid_50_path, 'r') as f:
        data = json.load(f)
        for _, v in data.items():
            total += v['elapsed_time']
            temp.append((v['elapsed_time']))
    
    for i, time in enumerate(temp):
        hybrid_timers[i+1] = time
    
    hybrid_avg_t = total / 150
    print(f'Hybrid AVG inference time: {hybrid_avg_t:.2f} seconds')

    # print(gpt_timers)
    df_gpt = pd.DataFrame(gpt_timers.items(), columns=['id', 'time'])
    df_hybrid = pd.DataFrame(hybrid_timers.items(), columns=['id', 'time'])

    combined_df = pd.merge(df_gpt, df_hybrid, on='id', suffixes=('_gpt', '_hybrid'))

    combined_df['time_hybrid'] = combined_df['time_hybrid'].round(2)
    

    avg_time_gpt = combined_df['time_gpt'].mean()
    avg_time_hybrid = combined_df['time_hybrid'].mean()

    avg_df = pd.DataFrame({'id': ['AVERAGE'], 'time_gpt': [avg_time_gpt], 'time_hybrid': [avg_time_hybrid]})

    # Concatenate the original DataFrame with the averages DataFrame
    combined_df = pd.concat([combined_df, avg_df], ignore_index=True)

    combined_df['time_gpt'] = combined_df['time_gpt'].round(2)
    combined_df['time_hybrid'] = combined_df['time_hybrid'].round(2)

    # combined_df.to_markdown('../result/time_eval.md', index=False)
    markdown_table = combined_df.to_markdown(buf=None, mode='pipe', index=True)
    with open('../result/time_eval.md', 'w') as file:
        file.write("## Benchmark GPT-4 only vs GPT-4 + ADA inference time\n\n")
        file.write(markdown_table)
    
    with open('time_eval.csv', 'w') as file:
        combined_df.to_csv(file, index=False)

if __name__ == '__main__':

    extract_time()