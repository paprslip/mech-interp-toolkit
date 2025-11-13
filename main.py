import torch
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
from toolkit import logit_lens, project_to_vocab, apply_steering, steering_vector, pca
from utils import load_data, load_tensors, save_tensors
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
import os
import matplotlib.pyplot as plt

load_dotenv(os.getcwd() + "/.env")
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATA_PATH = "./data/sample2.jsonl"
ALPHA = 5.0
HOOK_POINT = "resid_post"
BATCH_SIZE = 5

@torch.inference_mode()
def generate_steering_vector(model_name: str, data_path: str, alpha: int, hook_point: str, batch_size: int):
    print("="*60)
    print("STEERING VECTOR EXPERIMENT")
    print("="*60)
    # load model_name LLM onto GPU or CPU
    model = HookedTransformer.from_pretrained(model_name, device=DEVICE)
    # load a contrastive data set
    data = load_data(data_path)
    # default prompts
    prompts = data["prompts"]
    # prompt + positive completion
    positive = data["positive"]
    # prompt + negative completion
    negative = data["negative"]

    steering_vectors = {}
    steered_logits = {}
    # apply steering vectors at each layer and measure their effect
    for i in tqdm(range(model.cfg.n_layers), desc="Steering Vector Experiment", unit="layer"):
        layer = i
        steer_vec = steering_vector(
            model=model, 
            positive=positive, 
            negative=negative, 
            layer=layer, 
            hook_point=hook_point, 
            token_pos=-1,
            batch_size=batch_size
        )
        steering_vectors[f"layer.{i}.{hook_point}"] = steering_vector

        steered_logit = apply_steering(
            model=model, 
            prompts=prompts, 
            layer=layer, 
            steering_vector=steer_vec, 
            alpha=alpha, 
            hook_point=hook_point, 
            token_pos=-1, 
            batch_size=batch_size
        )
        steered_logits[f"layer.{i}.{hook_point}"] = steered_logit

    return (steering_vectors, steered_logits)


def capital_of_france(model_name: str, data_path: str, hook_point: str):
    baseline_logits = model(model.to_tokens(prompts))

    target_tokens = [12366, 38269, 60704,  6652,  7295]
    labels = []
    for token in target_tokens:
        labels.append(model.to_string(token))

    logits = logit_lens(model, model.to_tokens(prompts), [hook_point])
    mat = np.zeros((len(target_tokens), model.cfg.n_layers))

    print("TOP 5 PREDICTIONS")
    for i in range(model.cfg.n_layers):
        probs = torch.softmax(logits[f"layer.{i}.{hook_point}"], dim=-1)
        vals, idx = torch.topk(probs[0,-1,:], 5, dim=-1)
        print(model.to_string(idx), idx, vals)
        print(target_tokens)
        for j,token in enumerate(target_tokens):
            mat[j,i] = probs[0,-1,token]
    
    plt.figure(figsize=(len(target_tokens), model.cfg.n_layers))
    import matplotlib.colors as mcolors
    im = plt.imshow(mat, aspect="auto", origin="lower", cmap="plasma", norm=mcolors.LogNorm(vmin=mat.min(), vmax=mat.max()))
    plt.colorbar(im, label="Probability")
    plt.yticks(range(len(target_tokens)), labels, rotation=45, ha="right")
    plt.xticks(range(model.cfg.n_layers), range(model.cfg.n_layers))
    plt.ylabel("Token")
    plt.xlabel("Layer")
    plt.title("Evolution of token probabilities in baseline")
    plt.show()

    out = model.generate(model.to_tokens(prompts), max_new_tokens=100, temperature=0.0, top_k=None, top_p=None, do_sample=False)
    print(model.to_string(out))

    print("DONE")


if __name__ == "__main__":
    data = generate_steering_vector(MODEL_NAME, DATA_PATH, ALPHA, HOOK_POINT, BATCH_SIZE)
    save_tensors(list(data[0].values()), "./results/llama3-1_steering_vectors.safetensors", name=list(data[0].keys))
    save_tensors(list(data[1].values()), "./results/llama3-1_steered_logits.safetensors", name=list(data[1].keys))