from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch
import json
from torch.utils.data import DataLoader
from datasets import Dataset
import pickle
from itertools import chain

def read_dataset(filename):
    texts = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
    return texts

def tokenize_dataset(dataset):
    tokenized=[]
    for item in dataset:
        tokens = tokenizer(item, return_tensors="pt")
        tokenized.append(tokens)
    return tokenized

def chunk_dataset(tokenized_dataset, block_size):
    concatenated_input_ids = torch.cat([tokens['input_ids'][0] for tokens in tokenized_dataset], dim=0)
    concatenated_attention_masks = torch.cat([tokens['attention_mask'][0] for tokens in tokenized_dataset], dim=0)
    total_length = len(concatenated_input_ids)
    chunks = {
        'input_ids': [concatenated_input_ids[i : i + block_size] for i in range(0, total_length, block_size)],
        'attention_mask': [concatenated_attention_masks[i : i + block_size] for i in range(0, total_length, block_size)]
    }
    #Add padding for last chunk if less than block_size
    if len(chunks['input_ids'][-1]) < block_size:
        padding_length = block_size - len(chunks['input_ids'][-1])
        chunks['input_ids'][-1] = torch.cat([chunks['input_ids'][-1], torch.zeros(padding_length, dtype=torch.long)]) #pad_token_id is 0
        chunks['attention_mask'][-1] = torch.cat([chunks['attention_mask'][-1], torch.zeros(padding_length, dtype=torch.long)])
    return chunks

def batch_chunks(chunked_dataset, batch_size):
    total_chunks =  len(chunked_dataset['input_ids'])
    batched_chunks = []
    for i in range(0, total_chunks, batch_size):
        batched_chunks.append({
            'input_ids': torch.stack(chunked_dataset['input_ids'][i : i + batch_size]),
            'attention_mask': torch.stack(chunked_dataset['attention_mask'][i : i + batch_size])
        })
    return batched_chunks

def get_hidden_states(batch, model):
    outputs = model(batch, 
                    output_hidden_states = True,
                    use_cache = True, 
                    return_dict=True)
    hidden_states = outputs.cache_params  
    return hidden_states

model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = MambaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model.cuda()

raw_dataset = read_dataset("train_subset.jsonl")
tokenized_dataset = tokenize_dataset(raw_dataset)
chunked_dataset=chunk_dataset(tokenized_dataset, 128)
batched_chunks = batch_chunks(chunked_dataset, 4) #batched_chunks[batch_number]['input_ids'][instance_number]

encoded_dataset = batched_chunks.copy()
for idx, batch in enumerate(batched_chunks):
    print(idx)
    with torch.no_grad():
        input_ids = batch['input_ids'].to('cuda')
        hidden_states = get_hidden_states(input_ids, model)
        encoded_dataset[idx]['cache_params'] = hidden_states
    del hidden_states, input_ids
    torch.cuda.empty_cache()

# Save encoded_dataset as pickle file
with open("encoded_training_subset.pkl", "wb") as f:
    pickle.dump(encoded_dataset, f)