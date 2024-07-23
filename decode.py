import pickle
from transformers import MambaForCausalLM, AutoTokenizer
import torch

filename='encoded_training_subset.pkl'
with open(filename, 'rb') as f:
    train_data = pickle.load(f)

# Load the checkpoint
model1=MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
tokenizer = AutoTokenizer.from_pretrained('state-spaces/mamba-130m-hf')

model = MambaForCausalLM.from_pretrained('model/step_260', config = model1.config)
bos_inputs = tokenizer.batch_encode_plus([tokenizer.bos_token]*4, return_tensors='pt').to('cuda')

batch = train_data[0]
inp = batch['input_ids'].to('cuda')
cache = batch['cache_params']
model.cuda()


preds=[]

chunk_size=128
for idx in range(chunk_size):
    print(idx)
    if idx == 0:
        inputs = bos_inputs['input_ids']
        cache_params = cache
    else:
        inputs = next_tokens
    outputs = model(input_ids = inputs,
                        cache_params = cache_params,
                        use_cache=True,
                        output_dict=True,)
    cache_params = outputs.cache_params
    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1).view(-1, 1)
    preds.append(next_tokens)

gen = torch.cat(preds, dim=1)

for i in range(4):
    print('Input:', tokenizer.decode(inp[i], skip_special_tokens=True))
    print('Output:', tokenizer.decode(gen[i], skip_special_tokens=True))
    print('---------------------')
    print('\n')