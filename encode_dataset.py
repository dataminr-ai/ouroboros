from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch
import json
from torch.utils.data import DataLoader
from datasets import Dataset
import pickle

def get_hidden_states(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model(inputs["input_ids"], output_hidden_states = True)
    hidden_states = outputs.cache_params  
    return hidden_states

def read_dataset(filename):
    texts = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
    return texts

def encode_dataset(dataset):
    hidden_states=[]
    labels=[]
    for idx, text in enumerate(dataset):
        print(idx)
        hidden_state= get_hidden_states(text[:50], model, tokenizer)
        hidden_states.append(hidden_state)
        labels.append(text[:50])
    encoded_dataset = {'cache_params': hidden_states, 'labels': labels}
    return encoded_dataset

class EncodedDataset(Dataset):
    def __init__(self, encoded_dataset):
        self.encodings = encoded_dataset["cache_params"]
        self.labels = encoded_dataset["labels"]
    def __getitem__(self, idx):
        item = {}
        item["cache_params"] = self.encodings[idx]
        item["labels"] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = MambaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model.cuda()

dataset = read_dataset("train_subset.jsonl")
encoded_dataset = encode_dataset(dataset)

# Save encoded_dataset as pickle file
with open("encoded_training_subset.pkl", "wb") as f:
    pickle.dump(encoded_dataset, f)

#########################
##### Batch Encode ######
#########################
'''
from accelerate import Accelerator, DistributedType

accelerator = Accelerator()
text_column_name = 'text'
'''

block_size=model.config.max_position_embeddings

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


