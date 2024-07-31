import datasets
import pickle
from transformers import AutoConfig, MambaForCausalLM, AutoTokenizer
import torch
import os
import gc
import json
import re
import argparse

def move_cache(cache_params, device):
    with torch.no_grad():
        for key in cache_params.ssm_states:
            cache_params.ssm_states[key]=cache_params.ssm_states[key].to(device)
        for key in cache_params.conv_states:
            cache_params.conv_states[key]=cache_params.conv_states[key].to(device)
    torch.cuda.empty_cache()
    gc.collect()
    return cache_params

def reconstruct(model, tokenizer, batch, chunk_size=32, batch_size=20):
    inp = batch['input_ids'].to('cuda')
    cache = batch['cache_params']
    #cache.device='cuda'
    cache = move_cache(cache, 'cuda')
    cache.seqlen_offset = 0
    preds=[]
    for idx in range(chunk_size):
        #print(idx)
        if idx == 0:
            inputs = inp[:, 0].unsqueeze(1)
            cache_params = cache
        else:
            inputs = next_tokens
        with torch.no_grad():
            outputs = model(input_ids = inputs,
                            cache_params = cache_params,
                            use_cache=True,
                            output_dict=True,)
        cache_params = outputs.cache_params
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1).view(-1, 1)
        preds.append(next_tokens.to('cpu'))
    gen = torch.cat(preds, dim=1).to('cpu')
    recons=[]
    for i in range(batch['input_ids'].shape[0]):
        recons.append(tokenizer.decode(gen[i]))  
    del inp, inputs, cache, cache_params, outputs, next_tokens, gen, preds
    torch.cuda.empty_cache()
    gc.collect()
    return recons

def extract_step_number(path):
    match = re.search(r'step_(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("No step number found in the path")
    
def main(base_model, config_path, data_path, output_dir, ckpt_path=None):

    with open(data_path, 'rb') as f:
        eval = pickle.load(f)

    # Load the checkpoint
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    config = AutoConfig.from_pretrained(config_path)

    #Decode eval inputs
    decoded_inputs=[]
    for idx, batch in enumerate(eval):
        decoded_inputs.append(tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))

    ##################################
    ##### PREDICT AT CHECKPOINTS #####
    ##################################
    metric = datasets.load_metric('rouge')

    if not ckpt_path:
        model = MambaForCausalLM.from_pretrained(base_model, config=config)
        midx=0
    else:
        model = MambaForCausalLM.from_pretrained(ckpt_path, config=config)
        midx = extract_step_number(ckpt_path)
    model.cuda()
    model.eval()

    reconstructed=[]
    for idx, batch in enumerate(eval):
        print('Idx, ', idx)
        recons=reconstruct(model, tokenizer,batch)
        reconstructed.append(recons)
        del batch, recons
        torch.cuda.empty_cache()
        gc.collect()
        print(torch.cuda.memory_allocated())

    output_file=os.path.join(output_dir, str(midx)+'.pkl')
    with open(output_file, "wb") as f:
        pickle.dump(reconstructed, f) 

    del model
    torch.cuda.empty_cache()
    gc.collect()

    for idx, batch in enumerate(decoded_inputs):
        print(idx)
        pred=reconstructed[idx]
        metric.add(predictions=[pred], references=[batch])

    score=metric.compute()

    json_data = json.dumps(score, indent=4)
    output_file=os.path.join(output_dir, str(midx)+'.json')
    with open(output_file, 'w') as file:
        file.write(json_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script for a pre-trained model')
    parser.add_argument('--base_model', type=str, required=True, help='Base model name or path')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model config file')
    parser.add_argument('--data_path', type=str, required=True, help='Data for inference')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the outputs')
    parser.add_argument('--ckpt_path', type=str, required=False, help='Path to the checkpoint file')
    
    args = parser.parse_args()
    
    main(args.base_model, args.config_path, args.data_path, args.output_dir, args.ckpt_path,)