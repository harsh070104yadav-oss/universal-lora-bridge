import {
  cosine,
  frob,
  fromFactors,
  matMul,
  orthonormalColumns,
  relFrob,
  scale,
  scaleCols,
  svd,
  takeCols,
  takeRows,
  transpose,
  zeros,
  type Mat,
} from "./linalg.ts";
import {
  benchWidth,
  modelById,
  UPCOMING_ID,
  upcomingModel,
  type AttnKind,
  type CatalogModel,
  type ModelKind,
  type UpcomingSpec,
} from "./models.ts";

export type ModuleBlock = "attn" | "mlp" | "custom";

export const ROLES: Record<string, { role: string; block: ModuleBlock }> = {
  q_proj: { role: "attn_q", block: "attn" },
  k_proj: { role: "attn_k", block: "attn" },
  v_proj: { role: "attn_v", block: "attn" },
  o_proj: { role: "attn_o", block: "attn" },
  gate_proj: { role: "mlp_gate", block: "mlp" },
  up_proj: { role: "mlp_up", block: "mlp" },
  down_proj: { role: "mlp_down", block: "mlp" },
  q_a_proj: { role: "attn_q", block: "attn" },
  q_b_proj: { role: "attn_q_b", block: "attn" },
  kv_a_proj_with_mqa: { role: "attn_kv", block: "attn" },
  kv_b_proj: { role: "attn_kv_b", block: "attn" },
  "mlp.gate_proj": { role: "mlp_gate", block: "mlp" },
  "mlp.up_proj": { role: "mlp_up", block: "mlp" },
  "mlp.down_proj": { role: "mlp_down", block: "mlp" },
};

const ROLE_FALLBACK: Record<string, string[]> = {
  attn_q: ["attn_q", "attn_q_b"],
  attn_q_b: ["attn_q_b", "attn_q"],
  attn_k: ["attn_k", "attn_kv", "attn_kv_b"],
  attn_v: ["attn_v", "attn_kv", "attn_kv_b"],
  attn_kv: ["attn_kv", "attn_k+attn_v", "attn_kv_b"],
  attn_kv_b: ["attn_kv_b", "attn_kv", "attn_k+attn_v"],
  attn_o: ["attn_o"],
  mlp_gate: ["mlp_gate"],
  mlp_up: ["mlp_up"],
  mlp_down: ["mlp_down"],
};

export type Verdict = "transferable" | "bridged" | "partial" | "blocked";

export type ModuleReport = {
  sourceModules: string[];
  targetModule: string;
  role: string;
  mix: "direct" | "fused" | "expanded" | "generic";
  status: "transferred" | "unmatched";
  reason: string;
  capture: number;
  coeffResidual: number;
  spectraCosine: number;
  targetShape: [number, number] | null;
};

export type BridgeReport = {
  source: CatalogModel;
  target: CatalogModel;
  score: number;
  verdict: Verdict;
  parts: { arch: number; dim: number; coverage: number; spectral: number };
  modules: ModuleReport[];
  warnings: string[];
  elapsedMs: number;
  rank: number;
  alignRank: number;
  seed: number;
  bench: { sourceWidth: number; targetWidth: number };
};

export type Check = { name: string; pass: boolean; detail: string };

export type BridgeOptions = {
  sourceId: string;
  targetId: string;
  rank: number;
  alignRank: number;
  seed: number;
  upcoming?: UpcomingSpec;
};

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashSeed(seed: number, label: string): number {
  let h = seed >>> 0;
  for (let i = 0; i < label.length; i++) h = Math.imul(h ^ label.charCodeAt(i), 0x01000193);
  return h >>> 0;
}

export function roleOf(module: string): { role: string; block: ModuleBlock } {
  return ROLES[module] ?? { role: `custom:${module}`, block: "custom" };
}

export function moduleShape(module: string, width: number): { din: number; dout: number } {
  const role = roleOf(module).role;
  const wide = Math.min(48, width * 2);
  if (role === "mlp_gate" || role === "mlp_up") return { din: width, dout: wide };
  if (role === "mlp_down") return { din: wide, dout: width };
  if (role === "attn_q_b" || role === "attn_kv_b") {
    return { din: Math.max(8, Math.round(width / 2)), dout: width };
  }
  if (role === "attn_kv") return { din: width, dout: Math.max(8, Math.round(width / 2)) };
  if (role.startsWith("custom:")) return { din: width, dout: width };
  return { din: width, dout: width };
}

function modulesWithRole(model: CatalogModel, role: string): string[] {
  return model.modules.filter((name) => roleOf(name).role === role);
}

/** Which source LoRAs are rewritten into one target matrix. */
export function sourcesFor(role: string, source: CatalogModel): { modules: string[]; mix: ModuleReport["mix"] } {
  if (role.startsWith("custom:")) {
    const mlp = source.modules.filter((name) => roleOf(name).block === "mlp");
    const picked = mlp.length ? mlp : source.modules.slice(0, 1);
    return { modules: picked, mix: "generic" };
  }
  const ladder = ROLE_FALLBACK[role] ?? [role];
  for (const step of ladder) {
    if (step.includes("+")) {
      const parts = step.split("+").flatMap((part) => modulesWithRole(source, part));
      if (parts.length >= 2) return { modules: parts, mix: "fused" };
      continue;
    }
    const hit = modulesWithRole(source, step);
    if (!hit.length) continue;
    return { modules: hit, mix: step === role ? "direct" : hit.length > 1 ? "fused" : "expanded" };
  }
  return { modules: [], mix: "direct" };
}

export function archScore(source: CatalogModel, target: CatalogModel): number {
  if (source.id === target.id) return 1;
  if (source.family === target.family && source.kind === target.kind && source.attn === target.attn) return 0.92;
  if (source.kind === "dense" && target.kind === "dense" && source.attn === target.attn) return 0.78;
  if (source.kind === "dense" && target.kind === "dense") return 0.62;
  if (source.attn === "kda" || target.attn === "kda") return 0.34;
  if (source.attn === "mla" || target.attn === "mla") return 0.48;
  if (source.kind === "moe" && target.kind === "moe") return 0.5;
  return 0.4;
}

export type Transplant = {
  delta: Mat;
  capture: number;
  coeffResidual: number;
  spectraCosine: number;
};

/**
 * Express ΔW in the source weight's principal subspace, then rebuild that
 * coefficient inside the target weight's principal subspace.
 */
export function transplantDelta(
  sourceW: Mat,
  targetW: Mat,
  deltaW: Mat,
  alignRank: number,
  outRank: number,
): Transplant {
  const mS = sourceW.length;
  const nS = sourceW[0]?.length ?? 0;
  const mT = targetW.length;
  const nT = targetW[0]?.length ?? 0;
  const k = Math.max(1, Math.min(alignRank, mS, nS, mT, nT));

  const sourceSvd = svd(sourceW, k);
  const targetSvd = svd(targetW, k);
  const kk = Math.min(sourceSvd.s.length, targetSvd.s.length);
  const us = takeCols(sourceSvd.u, kk);
  const vs = transpose(takeRows(sourceSvd.vt, kk));
  const ut = takeCols(targetSvd.u, kk);
  const vt = transpose(takeRows(targetSvd.vt, kk));

  const coeff = matMul(transpose(us), matMul(deltaW, vs));
  const projected = matMul(us, matMul(coeff, transpose(vs)));
  const capture = Math.min(1, frob(projected) / (frob(deltaW) || 1));

  const lifted = matMul(ut, matMul(coeff, transpose(vt)));
  const factorRank = Math.max(1, Math.min(outRank, kk, mT, nT));
  const factored = svd(lifted, factorRank);
  const delta = fromFactors(factored.u, factored.s, factored.vt);
  const coeffAfter = matMul(transpose(ut), matMul(delta, vt));

  return {
    delta,
    capture,
    coeffResidual: relFrob(coeffAfter, coeff),
    spectraCosine: cosine(sourceSvd.s, targetSvd.s),
  };
}

function synthesizePair(din: number, dout: number, rank: number, rng: () => number): { weight: Mat; delta: Mat } {
  const k = Math.min(dout, din, Math.max(rank + 2, 6));
  const u = orthonormalColumns(dout, k, rng);
  const v = orthonormalColumns(din, k, rng);
  const spectrum = Array.from({ length: k }, (_, i) => k - i);
  const weight = matMul(scaleCols(u, spectrum), transpose(v));
  const r = Math.min(rank, k);
  const energy = Array.from({ length: r }, (_, i) => 0.35 / (i + 1));
  const delta = matMul(scaleCols(takeCols(u, r), energy), transpose(takeCols(v, r)));
  return { weight, delta };
}

function reasonFor(mix: ModuleReport["mix"], role: string, targetAttn: AttnKind): string {
  if (mix === "generic") return "Unknown future matrix. Filled from the source MLP LoRAs.";
  if (mix === "fused" && role.startsWith("attn_kv")) {
    return "Key and value LoRAs fused, then rewritten into this KV latent.";
  }
  if (mix === "expanded" && (role === "attn_q" || role === "attn_q_b")) {
    return "Query LoRA expanded into this latent factor.";
  }
  if (mix === "expanded") return "Closest source role rewritten into this matrix.";
  if (targetAttn === "kda" && role.startsWith("attn_")) return "Qwen attention update rewritten into the KDA factor.";
  return "Same role, rebuilt in the target subspace.";
}

function resolveTarget(options: BridgeOptions): CatalogModel {
  if (options.targetId === UPCOMING_ID) {
    return upcomingModel(
      options.upcoming ?? { name: "Upcoming model", hidden: 8192, attn: "kda", kind: "latent-moe", extraModules: [] },
    );
  }
  return modelById(options.targetId);
}

function verdictFor(score: number, coverage: number, meanResidual: number): Verdict {
  if (coverage === 0) return "blocked";
  if (coverage >= 0.999 && meanResidual < 0.05) return score >= 0.72 ? "transferable" : "bridged";
  if (coverage < 0.5 && score < 0.4) return "blocked";
  return "partial";
}

export function runBridge(options: BridgeOptions): BridgeReport {
  const started = Date.now();
  const source = modelById(options.sourceId);
  const target = resolveTarget(options);
  const sourceWidth = benchWidth(source.hidden);
  const targetWidth = benchWidth(target.hidden);
  const rank = Math.max(1, Math.round(options.rank));
  const alignRank = Math.max(rank, Math.round(options.alignRank));
  const warnings: string[] = [];

  if (target.attn === "kda" || target.attn === "mla") {
    warnings.push(
      "Attention layout differs. Q/K/V LoRAs are rewritten into the target's latent factors (q_a, q_b, kv_a, kv_b) instead of being dropped.",
    );
  }
  if (source.kind !== target.kind) {
    warnings.push("The router and expert index are not inside a LoRA. Only the linear updates move.");
  }
  warnings.push(
    `Checked on a ${sourceWidth} / ${targetWidth} stand-in of hidden ${source.hidden} → ${target.hidden}. The same map applies to full weights.`,
  );

  const modules: ModuleReport[] = target.modules.map((targetModule) => {
    const meta = roleOf(targetModule);
    const planned = sourcesFor(meta.role, source);
    const targetShape = moduleShape(targetModule, targetWidth);
    if (!planned.modules.length) {
      return {
        sourceModules: [],
        targetModule,
        role: meta.role,
        mix: planned.mix,
        status: "unmatched",
        reason: "No source LoRA can feed this matrix.",
        capture: 0,
        coeffResidual: 1,
        spectraCosine: 0,
        targetShape: [targetShape.dout, targetShape.din],
      };
    }

    const weightRng = mulberry32(hashSeed(options.seed, `W:${targetModule}`));
    const targetWeight = synthesizePair(targetShape.din, targetShape.dout, rank, weightRng).weight;
    const pieces = planned.modules.map((sourceModule) => {
      const sourceShape = moduleShape(sourceModule, sourceWidth);
      const rng = mulberry32(hashSeed(options.seed, sourceModule));
      const pair = synthesizePair(sourceShape.din, sourceShape.dout, rank, rng);
      return transplantDelta(pair.weight, targetWeight, pair.delta, alignRank, rank);
    });
    const capture = pieces.reduce((sum, piece) => sum + piece.capture, 0) / pieces.length;
    const coeffResidual = Math.max(...pieces.map((piece) => piece.coeffResidual));
    const spectraCosine = pieces.reduce((sum, piece) => sum + piece.spectraCosine, 0) / pieces.length;

    return {
      sourceModules: planned.modules,
      targetModule,
      role: meta.role,
      mix: planned.mix,
      status: "transferred",
      reason: reasonFor(planned.mix, meta.role, target.attn),
      capture,
      coeffResidual,
      spectraCosine,
      targetShape: [targetShape.dout, targetShape.din],
    };
  });

  const transferred = modules.filter((row) => row.status === "transferred");
  const coverage = transferred.length / Math.max(1, modules.length);
  const spectral = transferred.reduce((sum, row) => sum + row.spectraCosine, 0) / Math.max(1, transferred.length);
  const meanResidual = transferred.reduce((sum, row) => sum + row.coeffResidual, 0) / Math.max(1, transferred.length);
  const dim = Math.min(source.hidden, target.hidden) / Math.max(source.hidden, target.hidden);
  const arch = archScore(source, target);
  const score = 0.45 * arch + 0.25 * dim + 0.2 * coverage + 0.1 * Math.max(0, spectral);
  const verdict = verdictFor(score, coverage, meanResidual);

  return {
    source,
    target,
    score,
    verdict,
    parts: { arch, dim, coverage, spectral },
    modules,
    warnings,
    elapsedMs: Date.now() - started,
    rank,
    alignRank,
    seed: options.seed,
    bench: { sourceWidth, targetWidth },
  };
}

export function adapterPayload(report: BridgeReport) {
  return {
    format: "bridge-adapter-v1",
    base_model_name_or_path: report.target.id === UPCOMING_ID ? report.target.name : report.target.id,
    source_model: report.source.id,
    peft_type: "LORA",
    r: report.rank,
    lora_alpha: report.rank * 2,
    target_modules: report.modules.filter((row) => row.status === "transferred").map((row) => row.targetModule),
    bridge: report.modules.map((row) => ({
      from: row.sourceModules,
      to: row.targetModule,
      mix: row.mix,
      status: row.status,
      capture: Number(row.capture.toFixed(4)),
      coeff_residual: Number(row.coeffResidual.toExponential(2)),
      shape: row.targetShape,
    })),
    score: Number(report.score.toFixed(4)),
    verdict: report.verdict,
    note: "Module map is the target architecture. Re-run the transplant on full weights before serving a checkpoint.",
  };
}

function lowRankInSubspace(seed: number): { weight: Mat; delta: Mat } {
  const rng = mulberry32(seed);
  const u = orthonormalColumns(12, 6, rng);
  const v = orthonormalColumns(10, 6, rng);
  const weight = matMul(scaleCols(u, [8, 6, 5, 3, 2, 1]), transpose(v));
  const delta = matMul(scaleCols(takeCols(u, 3), [0.4, 0.25, 0.1]), transpose(takeCols(v, 3)));
  return { weight, delta };
}

export function selfCheck(): Check[] {
  const checks: Check[] = [];

  const diag = svd(
    [
      [3, 0],
      [0, 1],
    ],
    2,
  );
  const recon = fromFactors(diag.u, diag.s, diag.vt);
  const diagError = relFrob(recon, [
    [3, 0],
    [0, 1],
  ]);
  checks.push({
    name: "SVD rebuilds a diagonal",
    pass: diagError < 1e-6 && diag.s[0] > diag.s[1],
    detail: `relative error ${diagError.toExponential(2)}`,
  });

  const same = lowRankInSubspace(7);
  const recovered = transplantDelta(same.weight, same.weight, same.delta, 6, 3);
  const recovery = relFrob(recovered.delta, same.delta);
  checks.push({
    name: "Same base recovers the update",
    pass: recovery < 1e-4 && recovered.capture > 0.98,
    detail: `relative error ${recovery.toExponential(2)}, capture ${recovered.capture.toFixed(4)}`,
  });

  const scaled = transplantDelta(same.weight, same.weight, scale(same.delta, 3), 6, 3);
  checks.push({
    name: "Transfer is linear in the update",
    pass: relFrob(scaled.delta, scale(same.delta, 3)) < 1e-4,
    detail: `relative error ${relFrob(scaled.delta, scale(same.delta, 3)).toExponential(2)}`,
  });

  const wiped = transplantDelta(same.weight, same.weight, zeros(same.delta.length, same.delta[0].length), 6, 3);
  checks.push({
    name: "A zero adapter stays zero",
    pass: frob(wiped.delta) < 1e-8,
    detail: `frobenius ${frob(wiped.delta).toExponential(2)}`,
  });

  const other = lowRankInSubspace(99);
  const moved = transplantDelta(same.weight, other.weight, same.delta, 6, 3);
  checks.push({
    name: "A new basis keeps the coefficient",
    pass: moved.coeffResidual < 1e-4 && moved.delta.length === other.weight.length,
    detail: `coefficient residual ${moved.coeffResidual.toExponential(2)}`,
  });

  const qwenK3 = runBridge({ sourceId: "qwen25-7b", targetId: "kimi-k3", rank: 4, alignRank: 8, seed: 3 });
  const k3Filled = qwenK3.modules.every((row) => row.status === "transferred" && row.coeffResidual < 1e-3 && row.capture > 0.95);
  const k3Targets = qwenK3.modules.map((row) => row.targetModule);
  checks.push({
    name: "Qwen LoRA fills every Kimi K3 module",
    pass: k3Filled && qwenK3.verdict === "bridged" && k3Targets.includes("q_a_proj") && k3Targets.includes("kv_a_proj_with_mqa") && k3Targets.includes("mlp.down_proj"),
    detail: `${qwenK3.modules.length} modules, verdict ${qwenK3.verdict}, max residual ${Math.max(...qwenK3.modules.map((row) => row.coeffResidual)).toExponential(2)}`,
  });

  const upcoming = runBridge({
    sourceId: "qwen25-7b",
    targetId: UPCOMING_ID,
    rank: 4,
    alignRank: 8,
    seed: 5,
    upcoming: { name: "Next-1", hidden: 8192, attn: "kda", kind: "latent-moe" as ModelKind, extraModules: ["vision_merger"] },
  });
  const extra = upcoming.modules.find((row) => row.targetModule === "vision_merger");
  checks.push({
    name: "An upcoming model with a new matrix still receives the LoRA",
    pass: upcoming.modules.every((row) => row.status === "transferred") && extra?.mix === "generic" && extra.capture > 0.95,
    detail: `${upcoming.target.name} hidden ${upcoming.target.hidden}, ${upcoming.modules.length} matrices, verdict ${upcoming.verdict}`,
  });

  const near = runBridge({ sourceId: "llama32-3b", targetId: "llama31-8b", rank: 4, alignRank: 8, seed: 1 });
  checks.push({
    name: "Same-family transfer ranks as transferable",
    pass: near.verdict === "transferable" && near.score > qwenK3.score,
    detail: `Llama ${near.score.toFixed(2)} (${near.verdict}) vs Qwen→K3 ${qwenK3.score.toFixed(2)} (${qwenK3.verdict})`,
  });

  return checks;
}
