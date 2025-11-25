import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
import numpy as np
from jaxtyping import Float
from collections.abc import Callable

@torch.inference_mode()
def patching(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    clean_tokens: Float[torch.Tensor, "pos"], 
    corrupt_tokens: Float[torch.Tensor, "pos"],
    layer_idxs: list[int],
    metric_fn: Callable[[
        Float[torch.Tensor, "pos d_model"], 
        Float[torch.Tensor, "pos d_model"],
        Float[torch.Tensor, "pos d_model"]
        ], float]
) -> Float[np.ndarray, "layer_idx pos"]:
    """
    Activation patching
    """
    clean_out = model(
        clean_tokens,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True
    )

    clean_states = clean_out.hidden_states
    clean_logits = clean_out.logits
    
    corrupt_logits = model(
        corrupt_tokens,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True
    ).logits

    seq_len = len(corrupt_tokens[0])
    scores = np.zeros((seq_len, len(layer_idxs)))

    for i, layer_idx in enumerate(layer_idxs):
        clean_layer = clean_states[layer_idx + 1]
        for token_pos in range(seq_len):

            patched_logits = patch_layer(
                model=model,
                corrupt_tokens=corrupt_tokens,
                layer_idx=layer_idx,
                token_pos=token_pos,
                clean_layer=clean_layer
            )
            scores[token_pos, i] = metric_fn(
                clean_logits, 
                corrupt_logits, 
                patched_logits
            )
    return scores

def get_layer(
    model: AutoModelForCausalLM, 
    layer_idx: int
) -> torch.nn.Module:    
    """
    Method to get the layers of the model

    GPT-2 style     : model.transformers.h
    LLaMa / DeepSeek: model.model.layers   
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    
@torch.inference_mode()
def patch_layer(
    model: AutoModelForCausalLM, 
    corrupt_tokens: Float[torch.Tensor, "pos"],
    layer_idx: int, 
    token_pos: int, 
    clean_layer: Float[torch.Tensor, "pos d_model"]
) -> Float[torch.Tensor, "pos d_model"]:
    """
    Patch the residual stream of the model at a specific layer and position
    """
    layer = get_layer(model=model, layer_idx=layer_idx)

    def hook(module, inputs, output):
        out = output.clone()
        out[:, token_pos, :] = clean_layer[:, token_pos, :]
        return out
    
    handle = layer.register_forward_hook(hook)

    output = model(
        corrupt_tokens,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True
    )
    handle.remove()
    return output.logits

@torch.inference_mode()
def patch_layer_kv(
    model: AutoModelForCausalLM, 
    corrupt_tokens: Float[torch.Tensor, "pos"],
    layer_idx: int, 
    token_pos: int, 
    clean_layer: Float[torch.Tensor, "pos d_model"],
    patch_kv: Float[torch.Tensor, "n_heads pos d_head"]
):
    """
    Patch the residual stream AND the KV cache at a specific layer and position
    """
    layer = get_layer(model=model, layer_idx=layer_idx)

    def hook(module, inputs, output):
        resid_stream, kv = output

        resid = resid_stream.clone()
        resid[:, token_pos, :] = clean_layer[:, token_pos, :]

        k, v = kv
        patch_k, patch_v = patch_kv
        patched_k = k.clone()
        patched_v = v.clone()
        patched_k[:, :, token_pos, :] = patch_k[:, :, token_pos, :]
        patched_v[:, :, token_pos, :] = patch_v[:, :, token_pos, :]

        return resid, (patched_k, patched_v)

    handle = layer.register_forward_hook(hook)

    output = model(
        corrupt_tokens,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True
    )

    handle.remove()

    return output.logits


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    MODEL_NAME = "meta-llama/Llama-3.2-1B"

    clean_prompt = """Jane initially has 5 apples. After buying 8 more apples, how many apples does she have? To solve this, we add 5 + 8 = 16. Thus, Jane has a total of 13 apples. The above reasoning is:"""

    corrupt_prompt = """Jane initially has 5 apples. After buying 8 more apples, how many apples does she have? To solve this, we add 5 + 8 = 13. Thus, Jane has a total of 13 apples. The above reasoning is:"""

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    clean_tokens = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corrupt_tokens = tokenizer(corrupt_prompt, return_tensors="pt")["input_ids"]

    layer_idxs = [i for i in range(model.config.num_hidden_layers)]

    invalid_token = tokenizer(" invalid", add_special_tokens=False)["input_ids"]
    valid_token = tokenizer(" valid", add_special_tokens=False)["input_ids"]

    #def ioi_metric(clean_logits, corrupt_logits, patched_logits):
    #    return (patched_logits[:, -1, invalid_token] - patched_logits[:, -1, valid_token]).item()
    def logit_diff(
        logits: Float[torch.Tensor, "batch pos d_model"], 
        answer_tokens: list[int],    # list of right/wrong tokens to compare difference 
        per_prompt: bool = False                                # avg diff or per prompt diff
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        answer_logits = logits[:, -1, answer_tokens]
        answer_logits_diff = answer_logits[:, 0] - answer_logits[:, 1]
        if per_prompt:
            return answer_logits_diff
        return answer_logits_diff.mean()
    
    answer_tokens = [invalid_token, valid_token]
    
    def ioi_metric(clean_logits, corrupt_logits, patched_logits):
        clean_logit_diff = logit_diff(clean_logits, answer_tokens).item()
        corrupted_logit_diff = logit_diff(corrupt_logits, answer_tokens).item()
        return (logit_diff(patched_logits, answer_tokens) - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

    scores = patching(
        model=model,
        tokenizer=tokenizer,
        clean_tokens=clean_tokens,
        corrupt_tokens=corrupt_tokens,
        layer_idxs=layer_idxs,
        metric_fn=ioi_metric
    )

    #scores = scores - np.mean(scores)

    plt.figure(figsize=(12,6))
    plt.imshow(scores, aspect="auto", cmap="RdBu")
    plt.colorbar(label="IOI Score")
    plt.xticks(ticks=np.arange(model.config.num_hidden_layers), labels=layer_idxs)
    plt.xlabel("Layer")
    tokens = tokenizer.convert_ids_to_tokens(clean_tokens[0].tolist())
    tokens = [token.replace("Ġ","") for token in tokens]
    plt.yticks(ticks=np.arange(len(tokens)), labels=tokens, rotation=0)
    plt.ylabel("Sequence Position")
    plt.title("Activation Patching Heatmap")
    plt.tight_layout()
    plt.show()