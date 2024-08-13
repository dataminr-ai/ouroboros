import torch
import ouroboros.encode_dataset as ed
from ouroboros.models import MambaDecoderForCausalLM, MambaDecoderConfig
from ouroboros.models.modeling_mamba_decoder import MambaDecoderMixer
from transformers import AutoTokenizer
import gc
import inspect
import functools
import sys

# Load tokenizer and model
base_model = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model)

model=MambaDecoderForCausalLM.from_pretrained(base_model, use_mambapy=True)

# Input
text="Jon Snow is the heir to the Iron Throne"
tokenized =tokenizer(text, return_tensors="pt")

def log_shapes_and_values(output_file):
    """
    A decorator to log all local variables and their shapes inside the forward method.
    Logs are written to the specified output file.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0] 
            # Set up a frame to capture local variables at each point in the function
            frame = None
            def trace(frame, event, arg):
                nonlocal func
                if event == 'call':
                    return trace
                elif event == 'return':
                    # Log all local variables when returning
                    local_vars = frame.f_locals
                    with open(output_file, 'a') as f:
                        f.write(f"\n--- {self.__class__.__name__} ---\n")
                        for name, value in local_vars.items():
                            if isinstance(value, torch.Tensor):
                                f.write(f"{name}: shape {value.shape}, head {value.flatten()[:5]}\n")
                            else:
                                f.write(f"{name}: {value}\n")
                return trace
            # Set the trace function
            sys.settrace(trace)
            result = func(*args, **kwargs)
            sys.settrace(None)  # Clear the trace
            return result
        return wrapper
    return decorator

def apply_decorator_to_model(model, target_class, decorator):
    """
    Recursively apply a decorator to the forward method of each instance of target_class within the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, target_class):
            original_forward = module.forward
            module.forward = decorator(original_forward)
            print(f"Decorator applied to: {name}")
            
apply_decorator_to_model(model, MambaDecoderMixer, log_shapes_and_values('forward_test.txt'))
outputs = model(input_ids=tokenized['input_ids'])

