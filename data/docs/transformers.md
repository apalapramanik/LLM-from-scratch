# Transformers

The transformer is the neural network architecture that powers nearly every modern language model — GPT, Llama, Qwen, Claude, all of them. It was introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al.

The key idea is **self-attention**. Instead of processing words one at a time like older RNNs, a transformer looks at every word in the input simultaneously and decides which other words each one should pay attention to. This is what lets the model understand context: when it sees "bank" in "river bank" vs "money in the bank", attention pulls in the surrounding words to disambiguate.

A transformer is built from stacked **layers**, each containing:
- **Multi-head attention** — multiple parallel attention computations that look at different relationships
- **Feed-forward network** — a small MLP applied to each position independently
- **Layer normalization** and **residual connections** — these stabilize training

The number of parameters in a model is mostly determined by the number of layers, the hidden dimension, and the vocabulary size. A 7B-parameter model typically has 32 layers and a hidden dimension of 4096.

Modern LLMs are **decoder-only** transformers — they only have the right half of the original encoder-decoder architecture. They predict the next token given all the previous tokens, which is why they're called *autoregressive*.
