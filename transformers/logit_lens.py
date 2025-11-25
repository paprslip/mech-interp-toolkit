import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from jaxtyping import Float
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

@torch.inference_mode()
def logit_lens(
    model: AutoModelForCausalLM,
    tokenier: AutoTokenizer,
    prompt: str,
    plot: Optional[bool] = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the top token at each layer/pos
    """
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    output = model(
        prompt_tokens,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True
    )
    logits = project_to_logits(model, output.hidden_states)

    seq_len = len(prompt_tokens[0])
    n_layers = model.config.num_hidden_layers

    logits_arr = np.zeros((n_layers, seq_len))
    tokens_arr = np.empty((n_layers, seq_len), dtype=object)

    for layer in range(n_layers):
        for token_pos in range(seq_len):
            value, index = logits[layer][:, token_pos, :].max(dim=-1)
            logits_arr[layer, token_pos] = value.item()
            tokens_arr[layer, token_pos] = tokenizer.convert_ids_to_tokens(index.item())

    if plot:
        plt.figure(figsize=(14,12))
        ax = sns.heatmap(
            logits_arr[::-1],
            annot=tokens_arr[::-1],
            fmt="",
            cmap="YlGnBu",
            cbar=True,
            linewidth=.5
        )
        xlabels = [tokenizer.convert_ids_to_tokens(t.item()) for t in prompt_tokens[0]]
        ax.set_xlabel("Token position")
        ax.set_ylabel("Layer")
        ax.set_title("Model's top token and its logit per layer")
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_yticklabels([f"h{i}_out" for i in range(n_layers, 0, -1)], rotation=0)

        plt.tight_layout()
        plt.show()
    
    return (logits_arr, tokens_arr)

@torch.inference_mode()
def logit_evolution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    targets: list[str],
    token_pos: int = -1,
    include_topk: Optional[bool] = False,
    topk_layer: Optional[int] = -1,
    k: Optional[int] = 5,
    plot: Optional[bool] = False
) -> None:
    """
    Track the change in logit values across layers
    """
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    target_tokens = [i[0] for i in tokenizer(targets, add_special_tokens=False)["input_ids"]] if len(targets) > 0 else []
    output = model(
        prompt_tokens,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True
    )
    logits = project_to_logits(model, output.hidden_states)

    if include_topk:
        topk = torch.topk(logits[topk_layer][:, token_pos, :], k, dim=-1).indices.tolist()[0]
        target_tokens.extend(topk)

    n_layers = model.config.num_hidden_layers
    logits_arr = np.zeros((n_layers, len(target_tokens)))

    for layer in range(n_layers):
        for j, target in enumerate(target_tokens):
            value = logits[layer][:, token_pos, target]
            logits_arr[layer, j] = value.item()

    if plot:
        plt.figure(figsize=(14,12))
        ax = sns.heatmap(
            logits_arr[::-1],
            annot=logits_arr[::-1],
            fmt=".2f",
            cmap="YlGnBu",
            cbar=True,
            linewidth=.5
        )
        xlabels = [tokenizer.convert_ids_to_tokens(t) for t in target_tokens]
        ax.set_xlabel("Target token")
        ax.set_ylabel("Layer")
        ax.set_title("Evolution of target tokens across layers at token position")
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_yticklabels([f"h{i}_out" for i in range(n_layers, 0, -1)], rotation=0)

        plt.tight_layout()
        plt.show()
    
    return logits_arr


def project_to_logits(
    model: AutoModelForCausalLM,
    hidden_states: list[Float[torch.Tensor, "batch pos d_model"]],
) -> Float[torch.Tensor, "batch pos d_vocab"]:
    """
    Project model's hidden states/resdiual stream to logits
    """
    layers = []
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        for layer in range(1, model.config.num_hidden_layers):
            layers.append(model.transformer.ln_f(hidden_states[layer]) @ model.lm_head.weight.T)
        layers.append(hidden_states[-1] @ model.lm_head.weight.T)
        return layers
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        for layer in range(1, model.config.num_hidden_layers):
            layers.append(model.model.norm(hidden_states[layer]) @ model.lm_head.weight.T)
        layers.append(hidden_states[-1] @ model.lm_head.weight.T)
        return layers

    assert len(layers) == 0, "No logits returned from project_to_logits()"

if __name__ == "__main__":
    MODEL = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt = "What is the capital of France? The capital of France is"
    #logit_lens(model, tokenizer, prompt, plot=True)
    logit_evolution(model, tokenizer, prompt, targets=[" London", " Berlin", " Washington", " Marseille"], include_topk=True, plot=True)