import torch
import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from jaxtyping import Int, Float

@torch.inference_mode()
def logit_lens(
    model: HookedTransformer, 
    tokens: Int[torch.Tensor, "batch pos"], 
    hook_point: str, 
) -> dict[str, Float[torch.Tensor, "batch pos d_vocab"]]:

    has_bu = hasattr(model, "b_U")
    results = {}
    for layer in range(model.cfg.n_layers):
        act_name = tl_utils.get_act_name(hook_point, layer)
        _, cache = model.run_with_cache(tokens, names_filter = [act_name])

        resid = cache[hook_point, layer]
        normalized = model.ln_final(resid)
        proj = normalized @ model.W_U

        if has_bu:
            proj += model.b_U
        
        results[f"layer.{layer}.{hook_point}"] = proj

    return results

def project_to_vocab(model: HookedTransformer, resid: Float[torch.Tensor, "batch pos d_model"]):
    if hasattr(model, "b_U"):
        return model.ln_final(resid) @ model.W_U + model.b_U
    return model.ln_final(resid) @ model.W_U
