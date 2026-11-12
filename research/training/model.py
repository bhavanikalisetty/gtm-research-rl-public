"""Pinned Qwen3.5 multimodal loader with text-only, language-only LoRA."""
from __future__ import annotations

import json
import time
from pathlib import Path
from .policy import DecisionTokens, prompt_ids, prompt_version


def read_config(path=None):
    return json.loads(Path(path or Path(__file__).with_name("config.json")).read_text())


def load_model(config: dict, adapter: str | None = None, training: bool = False):
    if config["backend"] == "unsloth":
        # Import before transformers/PEFT, as required by Unsloth patches.
        from unsloth import FastVisionModel
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen BF16 smoke/training requires a CUDA GPU with BF16 support")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA GPU must support BF16")
    from transformers import AutoTokenizer
    from transformers import Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model
    torch.manual_seed(config["seed"])
    if config["backend"] == "unsloth":
        model, processor = FastVisionModel.from_pretrained(
            model_name=config["model"], revision=config["revision"],
            max_seq_length=config["max_context_tokens"], dtype=torch.bfloat16,
            load_in_4bit=False, load_in_16bit=True, fast_inference=False,
            full_finetuning=False)
        tokenizer = getattr(processor, "tokenizer", processor)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            config["model"], revision=config["revision"], dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=training)
    elif training:
        if config["backend"] == "unsloth":
            model = FastVisionModel.get_peft_model(
                model, finetune_vision_layers=False, finetune_language_layers=True,
                finetune_attention_modules=True, finetune_mlp_modules=True,
                r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=0,
                bias="none", random_state=config["seed"], use_gradient_checkpointing="unsloth")
        else:
            # Fully qualified language-only module names avoid touching vision q/k/v.
            targets = [name for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Linear) and "language_model" in name
                       and name.rsplit(".", 1)[-1] in {
                           "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}]
            if not targets:
                raise RuntimeError("No Qwen3.5 language LoRA target modules found")
            model = get_peft_model(model, LoraConfig(r=config["lora_rank"],
                lora_alpha=config["lora_alpha"], lora_dropout=0, bias="none", target_modules=targets,
                task_type="CAUSAL_LM"))
    if training:
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n or "visual" in n or "vision" in n for n, p in trainable):
            raise RuntimeError("Expected only language LoRA parameters to be trainable")
        if config["backend"] != "unsloth":
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    return model, tokenizer


def generate(model, tokenizer, state: dict, config: dict, sample=False):
    import torch
    ids = prompt_ids(tokenizer, state, config["max_context_tokens"], config["max_new_tokens"], prompt_version(config))
    device = next(model.parameters()).device
    batch = torch.tensor([ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        result = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
            max_new_tokens=config["max_new_tokens"], do_sample=sample,
            **({"temperature": config.get("rl_sampling_temperature", 1.0), "top_p": 1.0, "top_k": 0} if sample else {}),
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    model.train(was_training)
    completion = result[0, len(ids):].tolist()
    text = tokenizer.decode(completion, skip_special_tokens=True)
    return DecisionTokens(ids, completion, text), {
        "input_tokens": len(ids), "output_tokens": len(completion),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3)}


def token_logps(model, decision: DecisionTokens):
    import torch
    ids = torch.tensor([decision.input_ids], device=next(model.parameters()).device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    # Chunk vocab logsumexp by sequence position to avoid a second full fp32 logits tensor.
    start = len(decision.prompt) - 1
    logits = out.logits[0, start:-1, :].float()
    targets = ids[0, start + 1:]
    return logits.gather(1, targets[:, None]).squeeze(1) - logits.logsumexp(-1)
