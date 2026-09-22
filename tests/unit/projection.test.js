import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  pca2Model,
  pcaPointToLatent,
  procrustesAlignWithTransform,
} from '../../src/math/projection.js';

test('o PCA e sua inversa preservam pontos contidos em um plano', () => {
  const samples = [
    [-3, 0, 5],
    [3, 0, 5],
    [0, -1, 5],
    [0, 1, 5],
  ];
  const pca = pca2Model(samples);
  pca.points.forEach((point, index) => {
    const reconstructed = pcaPointToLatent(point, pca, { c: 1, s: 0 });
    reconstructed.forEach((value, dimension) => {
      assert.ok(Math.abs(value - samples[index][dimension]) < 1e-5);
    });
  });
});

test('o alinhamento estabiliza rotações e sua inversa recupera o embedding', () => {
  const reference = [
    [-2, 0],
    [2, 0],
    [0, -1],
    [0, 1],
  ];
  const rotated = reference.map(([x, y]) => [-y + 5, x - 7]);
  const aligned = procrustesAlignWithTransform(rotated, reference);
  aligned.points.forEach((point, index) => {
    point.forEach((value, dimension) =>
      assert.ok(Math.abs(value - reference[index][dimension]) < 1e-10),
    );
    const recovered = pcaPointToLatent(
      point,
      { mean: [5, -7], v1: [1, 0], v2: [0, 1] },
      aligned,
    );
    assert.deepEqual(Array.from(recovered), rotated[index]);
  });
});

test('conjuntos vazios ou sem variância não produzem coordenadas inválidas', () => {
  assert.equal(pca2Model([]), null);
  const pca = pca2Model([
    [1, 1],
    [1, 1],
  ]);
  assert.ok(pca.points.flat().every(Number.isFinite));
  assert.deepEqual(pca.points, [
    [0, 0],
    [0, 0],
  ]);
});
