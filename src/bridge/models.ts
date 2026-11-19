export type ModelKind = "dense" | "moe" | "latent-moe";
export type AttnKind = "gqa" | "mha" | "mla" | "kda";

export type CatalogModel = {
  id: string;
  name: string;
  family: string;
  hidden: number;
  kind: ModelKind;
  attn: AttnKind;
  note: string;
  modules: string[];
};

export const DENSE_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
export const LATENT_MODULES = [
  "q_a_proj",
  "q_b_proj",
  "kv_a_proj_with_mqa",
  "kv_b_proj",
  "o_proj",
  "mlp.gate_proj",
  "mlp.up_proj",
  "mlp.down_proj",
];

export const UPCOMING_ID = "upcoming";

export const MODELS: CatalogModel[] = [
  {
    id: "qwen25-7b",
    name: "Qwen2.5 7B",
    family: "qwen",
    hidden: 3584,
    kind: "dense",
    attn: "gqa",
    note: "Typical LoRA source. Dense GQA + SwiGLU.",
    modules: DENSE_MODULES,
  },
  {
    id: "qwen25-1.5b",
    name: "Qwen2.5 1.5B",
    family: "qwen",
    hidden: 1536,
    kind: "dense",
    attn: "gqa",
    note: "Smaller Qwen. Same module names.",
    modules: DENSE_MODULES,
  },
  {
    id: "qwen3-moe",
    name: "Qwen3 30B-A3B",
    family: "qwen",
    hidden: 2048,
    kind: "moe",
    attn: "gqa",
    note: "Routed experts. The bridge moves linear LoRAs, not the router.",
    modules: DENSE_MODULES,
  },
  {
    id: "llama32-3b",
    name: "Llama 3.2 3B",
    family: "llama",
    hidden: 3072,
    kind: "dense",
    attn: "gqa",
    note: "Dense GQA + SwiGLU.",
    modules: DENSE_MODULES,
  },
  {
    id: "llama31-8b",
    name: "Llama 3.1 8B",
    family: "llama",
    hidden: 4096,
    kind: "dense",
    attn: "gqa",
    note: "Same family, wider hidden size.",
    modules: DENSE_MODULES,
  },
  {
    id: "gemma2-2b",
    name: "Gemma 2 2B",
    family: "gemma",
    hidden: 2304,
    kind: "dense",
    attn: "mha",
    note: "Multi-head attention, GeGLU.",
    modules: DENSE_MODULES,
  },
  {
    id: "mistral-7b",
    name: "Mistral 7B",
    family: "mistral",
    hidden: 4096,
    kind: "dense",
    attn: "gqa",
    note: "Dense GQA.",
    modules: DENSE_MODULES,
  },
  {
    id: "kimi-k2",
    name: "Kimi K2",
    family: "kimi",
    hidden: 7168,
    kind: "moe",
    attn: "mla",
    note: "MLA latents. Query and KV are split factors, not plain Q/K/V.",
    modules: LATENT_MODULES,
  },
  {
    id: "kimi-k3",
    name: "Kimi K3",
    family: "kimi",
    hidden: 7168,
    kind: "latent-moe",
    attn: "kda",
    note: "KDA + latent MoE. Qwen Q/K/V is rewritten into q_a, q_b, kv_a, kv_b.",
    modules: LATENT_MODULES,
  },
];

export function modelById(id: string): CatalogModel {
  const found = MODELS.find((model) => model.id === id);
  if (!found) throw new Error(`Unknown model ${id}`);
  return found;
}

export type UpcomingSpec = {
  name: string;
  hidden: number;
  attn: AttnKind;
  kind: ModelKind;
  extraModules: string[];
};

export function modulesForAttn(attn: AttnKind): string[] {
  return attn === "mla" || attn === "kda" ? LATENT_MODULES : DENSE_MODULES;
}

export function upcomingModel(spec: UpcomingSpec): CatalogModel {
  const hidden = Math.max(256, Math.min(131072, Math.round(spec.hidden) || 8192));
  const base = modulesForAttn(spec.attn);
  const extra = spec.extraModules.map((name) => name.trim()).filter((name) => name && !base.includes(name));
  return {
    id: UPCOMING_ID,
    name: spec.name.trim() || "Upcoming model",
    family: "custom",
    hidden,
    kind: spec.kind,
    attn: spec.attn,
    note: "Not in the catalog. The bridge still fills every named matrix.",
    modules: [...base, ...extra],
  };
}

/** Map a published hidden size into a bench width the browser can factor quickly. */
export function benchWidth(hidden: number): number {
  const t = Math.round(16 + ((hidden - 1536) / (7168 - 1536)) * 20);
  return Math.max(16, Math.min(36, t));
}
