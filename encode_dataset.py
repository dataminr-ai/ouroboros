from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch
import json

def get_hidden_states(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model(inputs["input_ids"], output_hidden_states = True)
    hidden_states = outputs.hidden_states[-1]  # Take the last hidden state
    return hidden_states

def read_dataset(filename):
    texts = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
    return texts

def encode_dataset(dataset):
    encoded_dataset=[]
    for idx, text in enumerate(dataset):
        print(idx)
        hidden_state= get_hidden_states(text[:50], model, tokenizer)
        encoded_dataset.append({
                                 "input_embeds": hidden_state.tolist(),
                                 "labels": text[:50]
                                })
    return encoded_dataset

model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = MambaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model.cuda()

dataset = read_dataset("train_subset.jsonl")
encoded_dataset = encode_dataset(dataset)

with open('encoded_train_subset.json', 'w') as f:
    json.dump(encoded_dataset, f, indent=4)