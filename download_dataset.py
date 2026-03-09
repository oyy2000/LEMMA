from datasets import load_dataset
import json
dataset = load_dataset("panzs19/LEMMA", split="train")

dataset_list = dataset.to_list()

with open('./data/dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_list, f, indent=4, ensure_ascii=False)