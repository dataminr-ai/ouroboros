from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
mamba = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
mamba.cuda()

text = "Hello, this is a test"

inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = mamba(inputs["input_ids"], output_hidden_states = True , use_cache = True, return_dict=True)
hidden_states = outputs.hidden_states[-1]
cache = outputs.cache_params

#Next Token
next_token_logits = outputs.logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
tokenizer.decode(next_token_id[0])

#Generate
outputs = mamba.generate(inputs["input_ids"], do_sample=False)
tokenizer.decode(outputs[0])

####################
### Hidden State ###
####################
outputs = mamba(inputs_embeds = hidden_states)
next_token_logits = outputs.logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
tokenizer.decode(next_token_id[0])

#############
### CACHE ###
#############
text = "New"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = mamba(input_ids=inputs["input_ids"], cache_params = cache, use_cache=True)
next_token_logits = outputs.logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
tokenizer.decode(next_token_id[0])

######################
### Cache Training ###
######################

inputs = tokenizer(tokenizer.bos_token, return_tensors="pt").to("cuda")

mamba.train()
outputs = mamba(input_ids=inputs["input_ids"], cache_params = cache, use_cache=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mamba.parameters(), lr=0.001)

labels = tokenizer("Label", return_tensors="pt").to("cuda")
optimizer.zero_grad()
outputs = mamba(input_ids=inputs["input_ids"], cache_params = cache, use_cache=True, labels=labels["input_ids"])
loss = outputs.loss
loss.backward()
optimizer.step()