function centerRows(matrix) {
  const n = matrix.length;
  const d = matrix[0].length;
  const mean = new Float64Array(d);
  for (const row of matrix) for (let j = 0; j < d; j++) mean[j] += row[j] / n;
  const centered = matrix.map((row) => {
    const out = new Float64Array(d);
    for (let j = 0; j < d; j++) out[j] = row[j] - mean[j];
    return out;
  });
  return { centered, mean };
}

function matVecCov(centered, v) {
  const n = centered.length;
  const d = v.length;
  const out = new Float64Array(d);
  for (const row of centered) {
    let dot = 0;
    for (let j = 0; j < d; j++) dot += row[j] * v[j];
    for (let j = 0; j < d; j++) out[j] += row[j] * dot;
  }
  const denom = Math.max(1, n - 1);
  for (let j = 0; j < d; j++) out[j] /= denom;
  return out;
}

function normalize(v) {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function topEigenvector(centered, orthogonalTo = null) {
  const d = centered[0].length;
  let v = new Float64Array(d);
  for (let i = 0; i < d; i++)
    v[i] = Math.sin(i * 1.7 + 0.3) + 0.1 * Math.cos(i * 0.37);
  v = normalize(v);

  for (let k = 0; k < 35; k++) {
    const w = matVecCov(centered, v);
    if (orthogonalTo) {
      const proj = dot(w, orthogonalTo);
      for (let i = 0; i < d; i++) w[i] -= proj * orthogonalTo[i];
    }
    v = normalize(w);
  }
  return v;
}

/** Projeta os vetores nos dois eixos de maior variância por iteração de potência. */
export function pca2Model(matrix) {
  if (!matrix.length) return null;
  const { centered, mean } = centerRows(matrix);

  const v1 = topEigenvector(centered, null);
  const v2 = topEigenvector(centered, v1);

  const points = centered.map((row) => [dot(row, v1), dot(row, v2)]);

  return {
    points,
    mean: Array.from(mean),
    v1: Array.from(v1),
    v2: Array.from(v2),
  };
}

function center2(points) {
  let mx = 0,
    my = 0;
  for (const p of points) {
    mx += p[0];
    my += p[1];
  }
  mx /= points.length || 1;
  my /= points.length || 1;
  return points.map((p) => [p[0] - mx, p[1] - my]);
}

/** Alinha a rotação com a época anterior, preservando as distâncias entre pontos. */
export function procrustesAlignWithTransform(points, reference) {
  const X = center2(points);

  if (!reference || reference.length !== points.length) {
    return { points: X, c: 1, s: 0 };
  }

  const Y = center2(reference);
  let a = 0,
    b = 0;

  for (let i = 0; i < X.length; i++) {
    const x = X[i][0],
      y = X[i][1],
      u = Y[i][0],
      v = Y[i][1];
    a += x * u + y * v;
    b += x * v - y * u;
  }

  const norm = Math.hypot(a, b) || 1;
  const c = a / norm,
    ss = b / norm;

  return {
    c,
    s: ss,
    points: X.map(([x, y]) => [c * x - ss * y, ss * x + c * y]),
  };
}

/** Reverte a rotação da visualização e reconstrói um vetor no plano do PCA. */
export function pcaPointToLatent([alignedX, alignedY], pca, { c, s }) {
  const pc1 = c * alignedX + s * alignedY;
  const pc2 = -s * alignedX + c * alignedY;
  return Float32Array.from(
    pca.mean,
    (mean, j) => mean + pc1 * pca.v1[j] + pc2 * pca.v2[j],
  );
}
