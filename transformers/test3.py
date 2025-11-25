import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # ----------------------------
    # 1. Setup
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Replace this with the exact model ID you’re using, e.g.:
    #   "meta-llama/Llama-3.2-1B"
    #   "meta-llama/Llama-3.2-1B-Instruct"
    model_name = "gpt2"  # <--- change if needed

    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # LLaMA-style models usually need a pad token set for some utilities;
    # for our simple forward pass it's not strictly required, but harmless.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    # ----------------------------
    # 2. Choose a prompt
    # ----------------------------
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\nPrompt: {prompt!r}")
    print(f"Tokenized length: {inputs['input_ids'].shape[-1]} tokens")

    # ----------------------------
    # 3. Forward pass with hidden states
    # ----------------------------
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple: (embedding_out, layer1_out, ..., layerN_out)
    hidden_states = outputs.hidden_states
    num_layers = model.config.num_hidden_layers
    assert len(hidden_states) == num_layers + 1, "Unexpected number of hidden states"

    # ----------------------------
    # 4. Helper to inspect top tokens
    # ----------------------------
    def top_tokens_from_logits(logits, k=5):
        """
        logits: [vocab_size]
        returns: list of (token_str, prob) pairs
        """
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k, dim=-1)
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
        return list(zip(top_tokens, top_probs.cpu().tolist()))

    # We’ll look at the last position in the prompt
    pos = -1

    # ----------------------------
    # 5. Apply logit lens at each layer
    # ----------------------------
    print("\n=== Logit lens predictions per layer (last position) ===")

    for layer_idx, hs in enumerate(hidden_states):
        # hs shape: [batch, seq_len, hidden_dim]
        h_last = hs[0, pos, :]              # [hidden_dim]
        logits_lens = model.lm_head(model.transformer.ln_f(h_last)) # [vocab_size]

        top = top_tokens_from_logits(logits_lens, k=5)

        label = "Embedding" if layer_idx == 0 else f"Block {layer_idx}"
        print(f"\n--- {label} ---")
        for tok, p in top:
            print(f"{tok!r:>12}: {p:.4f}")

    # ----------------------------
    # 6. Verification: logit lens vs model logits
    # ----------------------------
    # Model's own logits for the last position
    direct_logits_last = outputs.logits[0, pos, :]  # [vocab_size]

    # Logit lens from the final layer hidden state
    final_hidden_last = hidden_states[-1][0, pos, :]
    lens_logits_last = model.lm_head(model.transformer.ln_f(final_hidden_last))

    # Check numerical closeness
    diff = torch.max(torch.abs(direct_logits_last - lens_logits_last)).item()
    print("\n=== Verification: final layer lens vs model logits ===")
    print(f"Max absolute difference between logits: {diff:.6f}")

    # Check argmax (predicted token) is the same
    direct_pred_id = torch.argmax(direct_logits_last).item()
    lens_pred_id = torch.argmax(lens_logits_last).item()

    direct_pred_str = tokenizer.decode([direct_pred_id])
    lens_pred_str = tokenizer.decode([lens_pred_id])

    print(f"Model prediction token: {direct_pred_str!r} (id={direct_pred_id})")
    print(f"Lens prediction token : {lens_pred_str!r} (id={lens_pred_id})")

    if direct_pred_id == lens_pred_id:
        print("✅ Predicted tokens match.")
    else:
        print("⚠️ Predicted tokens differ!")

    # Optional: assert tight numerical matching
    if diff < 1e-5 and direct_pred_id == lens_pred_id:
        print("✅ Logit lens is numerically consistent with the model’s final output.")
    else:
        print("⚠️ Logit lens is NOT exactly matching; check model / settings.")

if __name__ == "__main__":
    main()
