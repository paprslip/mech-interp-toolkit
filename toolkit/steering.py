import torch
import transformer_lens.utils as tl_utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer, ActivationCache
from functools import partial
from jaxtyping import Int, Float

@torch.inference_mode()
def apply_steering(
    model: HookedTransformer, 
    prompts: list[str], 
    layer: int, 
    steering_vector: torch.Tensor, 
    alpha: float, 
    hook_point: str, 
    token_pos: int =-1, 
    batch_size: int =1
) -> Float[torch.Tensor, "batch pos d_model"]:
    """
    TBD
    """
    def hook(
        logits: Float[torch.Tensor, "batch pos d_model"], 
        hook: HookPoint, 
        steering_vector: Float[torch.Tensor, "batch pos d_model"], 
        alpha: float, 
        token_pos: int
    ) -> Float[torch.Tensor, "batch pos d_model"]:

        if token_pos is None:
            return logits + alpha * steering_vector
        logits[:, token_pos, :] = logits[:, token_pos, :] + alpha * steering_vector
        return logits

    tokens = model.to_tokens(prompts)
    temp_hook_fn = partial(hook, steering_vector=steering_vector, alpha=alpha, token_pos=token_pos)
    logits = model.run_with_hooks(tokens, fwd_hooks=[(tl_utils.get_act_name(hook_point, layer), temp_hook_fn)])
    return logits

def steering_vector(
    model: HookedTransformer,
    positive: list[str], 
    negative: list[str], 
    layer: int, 
    hook_point: str ="resid_post", 
    token_pos: int = -1,
    batch_size: int = 1
) -> Float[torch.Tensor, "batch pos d_model"]:
    """
    TBD
    """
    @torch.inference_mode()
    def resid_mean(
        model: HookedTransformer,
        prompts: list[str], 
        layer: int, 
        hook_point: str = "resid_pre", 
        token_pos:int = token_pos,
        batch_size:int = batch_size
    ) -> Float[torch.Tensor, "batch pos d_model"]:

        vec = torch.zeros(model.cfg.d_model, device=model.cfg.device)

        for i in range(0, len(prompts), batch_size):
            _, cache = model.run_with_cache(model.to_tokens(prompts))
            activations = cache[hook_point, layer]
            vec += activations[:, token_pos, :].sum(dim=0)

        return vec / len(prompts)

    pos_mean = resid_mean(model, positive, layer, hook_point, token_pos)
    neg_mean = resid_mean(model, negative, layer, hook_point, token_pos)

    v = pos_mean - neg_mean
    v /= (v.norm() + 1e-12)

    return v