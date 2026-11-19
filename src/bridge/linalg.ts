/** Small dense linear algebra for subspace LoRA transfer. Row-major matrices. */

export type Mat = number[][];

export function zeros(rows: number, cols: number): Mat {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

export function matMul(a: Mat, b: Mat): Mat {
  const rows = a.length;
  const cols = b[0]?.length ?? 0;
  const inner = b.length;
  const out = zeros(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < inner; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < cols; j++) out[i][j] += aik * b[k][j];
    }
  }
  return out;
}

export function transpose(a: Mat): Mat {
  const rows = a.length;
  const cols = a[0]?.length ?? 0;
  const out = zeros(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) out[j][i] = a[i][j];
  }
  return out;
}

export function sub(a: Mat, b: Mat): Mat {
  return a.map((row, i) => row.map((v, j) => v - b[i][j]));
}

export function scale(a: Mat, s: number): Mat {
  return a.map((row) => row.map((v) => v * s));
}

export function frob(a: Mat): number {
  let sum = 0;
  for (const row of a) for (const v of row) sum += v * v;
  return Math.sqrt(sum);
}

export function relFrob(a: Mat, b: Mat): number {
  return frob(sub(a, b)) / (frob(b) || 1);
}

export function takeCols(a: Mat, k: number): Mat {
  return a.map((row) => row.slice(0, k));
}

export function takeRows(a: Mat, k: number): Mat {
  return a.slice(0, k).map((row) => row.slice());
}

/** Columns of `factors` (n × k) scaled by `s`, times nothing — returns n × k. */
export function scaleCols(a: Mat, s: number[]): Mat {
  return a.map((row) => row.map((v, j) => v * (s[j] ?? 0)));
}

export function fromFactors(u: Mat, s: number[], vt: Mat): Mat {
  return matMul(scaleCols(u, s), vt);
}

export function cosine(a: number[], b: number[]): number {
  const n = Math.max(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / Math.sqrt(na * nb);
}

function jacobiEigen(input: Mat, sweeps = 24): { values: number[]; vectors: Mat } {
  const n = input.length;
  const s = input.map((row) => row.slice());
  const v = zeros(n, n);
  for (let i = 0; i < n; i++) v[i][i] = 1;

  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += s[p][q] * s[p][q];
    }
    if (off < 1e-28) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = s[p][q];
        if (Math.abs(apq) < 1e-18) continue;
        const app = s[p][p];
        const aqq = s[q][q];
        const tau = (aqq - app) / (2 * apq);
        const sign = tau >= 0 ? 1 : -1;
        const t = sign / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const rot = t * c;

        s[p][p] = c * c * app - 2 * rot * c * apq + rot * rot * aqq;
        s[q][q] = rot * rot * app + 2 * rot * c * apq + c * c * aqq;
        s[p][q] = 0;
        s[q][p] = 0;

        for (let r = 0; r < n; r++) {
          if (r === p || r === q) continue;
          const arp = s[r][p];
          const arq = s[r][q];
          const nrp = c * arp - rot * arq;
          const nrq = rot * arp + c * arq;
          s[r][p] = nrp;
          s[p][r] = nrp;
          s[r][q] = nrq;
          s[q][r] = nrq;
        }

        for (let r = 0; r < n; r++) {
          const vip = v[r][p];
          const viq = v[r][q];
          v[r][p] = c * vip - rot * viq;
          v[r][q] = rot * vip + c * viq;
        }
      }
    }
  }

  return { values: s.map((row, i) => row[i]), vectors: v };
}

export type Svd = { u: Mat; s: number[]; vt: Mat };

/** Thin SVD, components ordered by descending singular value. */
export function svd(a: Mat, rank?: number): Svd {
  const m = a.length;
  const n = a[0]?.length ?? 0;
  const full = Math.min(m, n);
  const keep = Math.max(0, Math.min(rank ?? full, full));
  if (m === 0 || n === 0 || keep === 0) {
    return { u: zeros(m, 0), s: [], vt: zeros(0, n) };
  }

  if (n <= m) {
    const eig = jacobiEigen(matMul(transpose(a), a));
    const order = eig.values
      .map((value, index) => ({ value, index }))
      .sort((p, q) => q.value - p.value);
    const s: number[] = [];
    const vt: Mat = [];
    const u = zeros(m, keep);
    for (let t = 0; t < keep; t++) {
      const sigma = Math.sqrt(Math.max(0, order[t].value));
      s.push(sigma);
      const right = eig.vectors.map((row) => row[order[t].index]);
      vt.push(right);
      if (sigma > 1e-12) {
        const av = matMul(a, right.map((x) => [x])).map((row) => row[0] / sigma);
        for (let r = 0; r < m; r++) u[r][t] = av[r];
      }
    }
    return { u, s, vt };
  }

  const eig = jacobiEigen(matMul(a, transpose(a)));
  const order = eig.values
    .map((value, index) => ({ value, index }))
    .sort((p, q) => q.value - p.value);
  const s: number[] = [];
  const vt: Mat = [];
  const u = zeros(m, keep);
  for (let t = 0; t < keep; t++) {
    const sigma = Math.sqrt(Math.max(0, order[t].value));
    s.push(sigma);
    const left = eig.vectors.map((row) => row[order[t].index]);
    for (let r = 0; r < m; r++) u[r][t] = left[r];
    if (sigma > 1e-12) {
      const at = matMul(transpose(a), left.map((x) => [x])).map((row) => row[0] / sigma);
      vt.push(at);
    } else {
      vt.push(Array(n).fill(0));
    }
  }
  return { u, s, vt };
}

/** Random n × k matrix with orthonormal columns. */
export function orthonormalColumns(n: number, k: number, rng: () => number): Mat {
  const raw = zeros(n, k);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < k; c++) raw[r][c] = rng() * 2 - 1;
  }
  for (let c = 0; c < k; c++) {
    for (let p = 0; p < c; p++) {
      let dot = 0;
      for (let r = 0; r < n; r++) dot += raw[r][c] * raw[r][p];
      for (let r = 0; r < n; r++) raw[r][c] -= dot * raw[r][p];
    }
    let norm = 0;
    for (let r = 0; r < n; r++) norm += raw[r][c] * raw[r][c];
    norm = Math.sqrt(norm) || 1;
    for (let r = 0; r < n; r++) raw[r][c] /= norm;
  }
  return raw;
}
