from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
model.cuda()

#####################
### Hidden States ###
#####################
text = "Hello, this is a test"

inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = model(inputs["input_ids"], 
                output_hidden_states = True , 
                use_cache = True, 
                return_dict=True)
cache = outputs.cache_params

#############
### Train ###
#############
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()

inputs = tokenizer(tokenizer.bos_token, return_tensors="pt").to("cuda")
labels = tokenizer("Label", return_tensors="pt").to("cuda")

model.train()
outputs = model(input_ids=inputs["input_ids"], 
                cache_params = cache, 
                use_cache=True, 
                labels=labels["input_ids"])
loss = outputs.loss
loss.backward()
optimizer.step()
