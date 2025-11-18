import torch
from sentence_transformers import SentenceTransformer

def infer(input, useless_arg_1, model, useless_arg_2):
    with torch.no_grad():
        return model.encode(input)

def load(device, llm_name):
    model = SentenceTransformer(llm_name)
    model.eval().to(device=device)
    return infer, None, model
