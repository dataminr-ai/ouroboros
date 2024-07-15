from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
mamba = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

outputs = model(**inputs)
hidden_states = outputs.hidden_states[-1]