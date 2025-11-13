import torch
from transformer_lens import HookedTransformer, ActivationCache
from collections.abc import Callable
import json
from pathlib import Path
from safetensors.torch import save_file, load_file

def load_data(path):
    questions = []
    pos_prompts = []
    neg_prompts = []
    
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["question"])
            pos_prompts.append(data["question"] + data["answer_matching_behavior"])
            neg_prompts.append(data["question"] + data["answer_not_matching_behavior"])

    return {"prompts": questions, "positive": pos_prompts, "negative": neg_prompts}

def save_tensors(tensors: list[torch.Tensor], path: str, name: list[str]):
    if len(tensors) != len(name):
        raise ValueError("Each tensor to be saved must have a key!")
    path = Path(path)
    data = {}
    for i in range(len(tensors)):
        data[name] = tensors[i]
    save_file({name: tennsor.detach().cpu()}, str(path))

def load_tensors(path: str) -> dict[str, torch.Tensor]:
    return load_file(path)