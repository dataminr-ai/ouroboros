import pickle
from transformers import AutoConfig, MambaForCausalLM, AutoTokenizer
import torch

filename='encoded_training_subset_32.pkl'
with open(filename, 'rb') as f:
    train_data = pickle.load(f)

# Load the checkpoint
tokenizer = AutoTokenizer.from_pretrained('state-spaces/mamba-130m-hf')
config = AutoConfig.from_pretrained('model/eval_config.json')
model = MambaForCausalLM.from_pretrained('model/model32_e6/epoch_0', config=config)

bos_inputs = tokenizer.batch_encode_plus([tokenizer.bos_token]*4, return_tensors='pt').to('cuda')


#Batch
batch = train_data[0]
inp = batch['input_ids'].to('cuda')
cache = batch['cache_params']
cache.seqlen_offset = 0
model.cuda()

preds=[]

chunk_size=32
for idx in range(chunk_size):
    print(idx)
    if idx == 0:
        inputs = inp[:, 0].unsqueeze(1)
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

for i in range(14):
    print('Text:', tokenizer.decode(inp[i]))
    print('\n')
    print('Reconstruction:', tokenizer.decode(gen[i]))
    print('\n\n')