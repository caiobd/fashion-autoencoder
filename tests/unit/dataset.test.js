import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import {
  FASHION_IMAGES_URL,
  FASHION_LABELS_URL,
  parseFashionMnist,
  selectSamples,
} from '../../src/data/fashion-mnist.js';

test('a cópia local preserva os arquivos originais e as dez mil imagens', async () => {
  const expectedHashes = [
    '346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073',
    '67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5',
  ];
  const files = await Promise.all(
    [FASHION_IMAGES_URL, FASHION_LABELS_URL].map(async (url, index) => {
      const compressed = await readFile(new URL(url));
      assert.equal(
        createHash('sha256').update(compressed).digest('hex'),
        expectedHashes[index],
        'Dataset ausente ou alterado. Confira os objetos com git lfs pull.',
      );
      const bytes = gunzipSync(compressed);
      return bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      );
    }),
  );
  const dataset = parseFashionMnist(...files);
  assert.equal(dataset.labels.length, 10000);
  assert.equal(dataset.pixels.length, 10000 * 784);
  assert.deepEqual(
    dataset.byClass.map((indices) => indices.length),
    Array(10).fill(1000),
  );
});

function idxFiles() {
  const count = 100;
  const images = new ArrayBuffer(16 + count * 784);
  const labels = new ArrayBuffer(8 + count);
  const imageHeader = new DataView(images);
  [2051, count, 28, 28].forEach((value, index) =>
    imageHeader.setUint32(index * 4, value),
  );
  const labelHeader = new DataView(labels);
  labelHeader.setUint32(0, 2049);
  labelHeader.setUint32(4, count);
  for (let index = 0; index < count; index++) {
    new Uint8Array(images, 16 + index * 784, 784).fill(index);
    new Uint8Array(labels)[8 + index] = index % 10;
  }
  return { images, labels };
}

test('seleciona as mesmas amostras alternadas e normalizadas do HTML original', () => {
  const { images, labels } = idxFiles();
  const dataset = parseFashionMnist(images, labels);
  const selected = selectSamples(dataset, 30);
  assert.deepEqual(
    selected.labels,
    Array.from({ length: 30 }, (_, index) => index % 10),
  );
  // Na segunda volta, (1 × 37) % 10 = 7: índice 70 da classe 0.
  assert.equal(selected.images[10][0], Math.fround(70 / 255));
  assert.equal(selected.images[20][783], Math.fround(40 / 255));
  assert.deepEqual(selected, selectSamples(dataset, 30));
});

test('rejeita cabeçalhos, dimensões, contagens e dados IDX inválidos', () => {
  assert.throws(
    () => parseFashionMnist(new ArrayBuffer(0), new ArrayBuffer(0)),
    /Cabeçalho/,
  );
  for (const [offset, value] of [
    [0, 7],
    [4, 99],
    [8, 32],
  ]) {
    const { images, labels } = idxFiles();
    new DataView(images).setUint32(offset, value);
    assert.throws(() => parseFashionMnist(images, labels));
  }
  const { images, labels } = idxFiles();
  assert.throws(
    () => parseFashionMnist(images.slice(0, -1), labels),
    /Tamanho/,
  );
  new Uint8Array(labels)[8] = 10;
  assert.throws(() => parseFashionMnist(images, labels), /Classe inválida/);
});

test('limita a quantidade e exige exemplos de todas as classes', () => {
  const { images, labels } = idxFiles();
  const dataset = parseFashionMnist(images, labels);
  assert.equal(selectSamples(dataset, 2000).images.length, 100);
  assert.equal(selectSamples(dataset, 1).images.length, 10);
  dataset.byClass[0] = [];
  assert.throws(() => selectSamples(dataset, 10), /dez classes/);
});
