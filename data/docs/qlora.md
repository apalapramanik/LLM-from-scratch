# QLoRA

QLoRA (Quantized Low-Rank Adaptation) is a fine-tuning method that lets you train large language models on a single consumer GPU. It combines two tricks: 4-bit quantization of the base model, and LoRA adapters for the actual training.

**The problem:** Full fine-tuning of a 7B model in FP16 needs ~28 GB of VRAM just for the weights, plus more for gradients and optimizer states. That doesn't fit on a single 24 GB GPU.

**The fix — 4-bit quantization:** Store the frozen base model weights in 4 bits instead of 16. This shrinks the weight memory by 4x. The bitsandbytes library implements a specific quantization scheme called NF4 (NormalFloat 4-bit) which preserves accuracy well for normally-distributed weights.

**The fix — LoRA adapters:** Instead of training every weight in the model, freeze the base weights and inject small low-rank matrices (typically rank 8 or 16) into the attention layers. Only these adapters get trained. For a 7B model this means training ~10M parameters instead of 7B — a 700x reduction.

**The result:** You can fine-tune a 7B model on a single 24 GB GPU, or even a 13B model on a 48 GB A6000. Training is slower than full fine-tuning but the quality is comparable for most tasks.

QLoRA is implemented in the HuggingFace `peft` library, with the quantization handled by `bitsandbytes`. The training loop itself is run via `transformers.Trainer` or the `trl` library's `SFTTrainer`.
