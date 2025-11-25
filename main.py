import torch
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
from toolkit import logit_lens, logit_diff, project_to_vocab, apply_steering, steering_vector, generate_steering, pca
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
DATA_PATH = "./data/sample3.jsonl"
ALPHA = 5.0
HOOK_POINT = "resid_post"
BATCH_SIZE = 20

@torch.inference_mode()
def generate_steering_vector(model_name: str, data_path: str, hook_point: str, batch_size: int):
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
        steering_vectors[f"layer.{i}.{hook_point}"] = steer_vec.to("cpu")
        del steer_vec
        torch.cuda.empty_cache()

    return steering_vectors

def test_steering_vector(model_name:str, data_path:str, steering_vec, layer: int, alpha: float, hook_point: str, batch_size: int):
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

    steered_logits = apply_steering(
        model=model, 
        prompts=prompts, 
        layer=layer, 
        steering_vector=steering_vec, 
        alpha=alpha, 
        hook_point=hook_point, 
        token_pos=-1, 
        batch_size=batch_size
    )
    
    baseline_logits = model(model.to_tokens(prompts))
    print(baseline_logits.shape)
    baseline_probs = torch.softmax(baseline_logits[0,-1,:], dim=-1)
    steered_probs = torch.softmax(steered_logits[0,-1,:], dim=-1)
    k = 20
    baseline_vals, baseline_idx = torch.topk(baseline_probs, k, dim =-1)
    steered_vals, steered_idx = torch.topk(steered_probs, k, dim=-1)

    print("**********Baseline Top k predictions**********")
    for i in range(k):
        print(model.to_string(baseline_idx[i]))
        print(baseline_vals[i])
    print("***********Steered Top k predictions**********")
    for i in range(k):
        print(model.to_string(steered_idx[i]))
        print(steered_vals[i])

    #print(generate_steering(model, prompts, 5, steering_vec, alpha, hook_point, max_new_tokens=128, temperature=0.0))

    return steered_logits

def capital_of_france(model_name: str, data_path: str, hook_point: str):
    baseline_logits = model(model.to_tokens(prompts))

    target_tokens = [12366, 38269, 60704,  6652,  7295]
    labels = []
    for token in target_tokens:
        labels.append(model.to_string(token))

    logits = logit_lens(model, model.to_tokens(prompts), hook_point)
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

@torch.inference_mode()
def activation_patching(model_name: str, data_path: str):
    import transformer_lens.patching as patching
    from neel_plotly import line, imshow, scatter

    model = HookedTransformer.from_pretrained(model_name, device=DEVICE)

    answers = [" Invalid", " Valid"]
    answer_tokens = torch.tensor([[model.to_single_token(answer) for answer in answers]])

    data = load_data(data_path)
    clean_tokens = model.to_tokens(data["prompts"][0])
    corrupted_tokens = model.to_tokens(data["prompts"][1])
    
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    clean_logit_diff = logit_diff(clean_logits, answer_tokens).item()
    corrupted_logit_diff = logit_diff(corrupted_logits, answer_tokens).item()

    def ioi_metric(logits, answer_tokens=answer_tokens):
        return (logit_diff(logits, answer_tokens) - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

    resid_pre_activation_patching = patching.get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, ioi_metric)

    imshow(resid_pre_activation_patching, yaxis="layer", xaxis="position",x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))])

if __name__ == "__main__":
    #data = generate_steering_vector(MODEL_NAME, DATA_PATH, HOOK_POINT, BATCH_SIZE)
    #save_tensors(tensors=list(data[0].values()), names=list(data[0].keys()), path="./results/llama3-1_steering_vectors.safetensors")
    #save_tensors(tensors=list(data[1].values()), names=list(data[1].keys()), path="./results/llama3-1_steered_logits.safetensors")
    #save_tensors(tensors=list(data.values()), names=list(data.keys()), path="./results/llama3-1_steering_vectors.safetensors")
    #data = load_tensors(path="./results/llama3-1_steering_vectors.safetensors")
    #rint(data)
    #steer_vec = data["layer.5.resid_post"]
    #test = test_steering_vector(MODEL_NAME, DATA_PATH, steer_vec, 5, ALPHA, HOOK_POINT, BATCH_SIZE)

    activation_patching(MODEL_NAME, DATA_PATH)