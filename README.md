# Universal LoRA Bridge

**Train a LoRA on any model. Use it on any other model.**

Experimental research project for transferring LoRA adapters across heterogeneous LLM architectures.

## Status

Early prototype / research implementation. Not production-ready.

## Motivation

Standard LoRA adapters are tightly coupled to the exact model they were trained on due to architecture and dimension mismatches. This project explores practical implementations of recent research that aims to break that limitation.

### Key Research

- **Cross-LoRA** (Xia et al., 2025) — Data-free, training-free transfer via SVD subspace alignment (arXiv:2508.05232)
- **Trans-LoRA** (Wang et al., 2024) — Synthetic data + distillation for near-lossless transfer (NeurIPS 2024)
- **LoRA-X** — Training-free cross-model adaptation
- **Universal-Adopter LoRA (UAL)** — Architecture-agnostic intermediate representation

## Goals

1. Working implementation of Cross-LoRA-style subspace alignment for common model families (Llama, Qwen, Gemma, etc.).
2. Support transferring LoRAs between models of different sizes.
3. Both data-free (weight-space) and data-assisted (synthetic data) modes.
4. Document limitations with highly divergent architectures (e.g. standard Transformer → Kimi K3-style KDA + Latent MoE).

## Planned Structure

```
universal-lora-bridge/
├── README.md
├── requirements.txt
├── transfer.py              # Main transfer script
├── src/
│   ├─── cross_lora.py       # Subspace alignment (SVD)
│   ├─── align.py            # Weight space alignment utilities
│   └─── utils.py
└── examples/
```

## Disclaimer

Transfer quality varies significantly with architectural similarity. Extreme mismatches are expected to perform poorly or fail. This is research software.

## License

Apache 2.0
