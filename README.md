# Universal LoRA Bridge

Rewrites a LoRA from one architecture into another’s matrices.

A Qwen2.5 LoRA is mapped onto Kimi K3 instead of being dropped:

| Qwen module | Kimi K3 module |
| --- | --- |
| `q_proj` | `q_a_proj` and `q_b_proj` |
| `k_proj` + `v_proj` | `kv_a_proj_with_mqa` and `kv_b_proj` |
| `o_proj` | `o_proj` |
| `gate_proj`, `up_proj`, `down_proj` | `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` |

An unknown future matrix is filled from the source MLP LoRAs.

## What this is

`src/bridge/` is the tested implementation.

- `transplantDelta` expresses the update in the source weight’s principal subspace and rebuilds that coefficient in the target subspace.
- `runBridge` plans every target module, including KDA / MLA layouts and extra names on an upcoming model.
- Same-family pairs (Llama → Llama) are marked `transferable`. Qwen → Kimi K3 is marked `bridged`.

The bench uses a scaled stand-in of each hidden size. The module map is the same one you would apply to full weights. This repository does not ship Kimi or Qwen checkpoints, and a bridged adapter is not a LoRA that was trained on the target.

## Check

Requires Node 22.

```bash
node --experimental-strip-types --test src/bridge/bridge.test.ts
```

Eight checks, including “Qwen LoRA fills every Kimi K3 module” and “an upcoming model with a new matrix still receives the LoRA.”

## License

Apache-2.0
