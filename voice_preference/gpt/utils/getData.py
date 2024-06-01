import json

def get_data() -> None:
    hashmap = {}
    with open("../data/new_generic.json", "r") as f:
        data = json.load(f)
    
    with open("../data/new_50.json", "r") as f:
        new = json.load(f)
    
    for k, v in data.items():
        hashmap[int(v['id'])] = k
    
    for k, v in new.items():
        hashmap[int(v['id']) + 100] = k
    
    with open("../data/overall_dataset.json", "w") as f:
        json.dump(hashmap, f, indent=4)


if __name__ == "__main__":
    get_data()