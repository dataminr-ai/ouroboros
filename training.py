from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch
import pandas as pd
import json
from torch.utils.data import Dataset

###############
### Dataset ###
###############
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
mamba = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

def get_hidden_states(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states[-1]  # Take the last hidden state
    return hidden_states

class EncodingsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

file_path = "train_subset.jsonl"
texts = []

with open(file_path, "r") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["text"])

hidden_states = [get_hidden_states(text, mamba, tokenizer) for text in texts]

encodings = {"input_ids": hidden_states}
dataset = EncodingsDataset(encodings, texts)

################
### Training ###
################
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)

lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    #dataset_text_field="quote",
)
trainer.train()
