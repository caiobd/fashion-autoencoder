import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import { createAutoencoderModels } from '../../src/model/architecture.js';

await tf.setBackend('cpu');
await tf.ready();

test('a definição treina sozinha e compartilha os pesos com encoder e decoder', async () => {
  const baseline = tf.memory().numTensors;
  const { encoder, decoder, model } = createAutoencoderModels(tf, 4);
  const optimizer = tf.train.adam(0.002);
  const pixels = tf.tensor2d(
    [Array.from({ length: 784 }, (_, index) => (index % 28) / 27)],
    [1, 784],
  );

  async function predict() {
    const [embedding, decoded, reconstructed] = tf.tidy(() => {
      const embedding = encoder.predict(pixels);
      return [embedding, decoder.predict(embedding), model.predict(pixels)];
    });
    try {
      assert.deepEqual(embedding.shape, [1, 4]);
      assert.deepEqual(decoded.shape, [1, 784]);
      assert.deepEqual(reconstructed.shape, [1, 784]);
      const full = await reconstructed.data();
      const parts = await decoded.data();
      full.forEach((value, index) => {
        assert.ok(Number.isFinite(value) && value >= 0 && value <= 1);
        assert.ok(Math.abs(value - parts[index]) < 1e-6);
      });
      return Array.from(full);
    } finally {
      embedding.dispose();
      decoded.dispose();
      reconstructed.dispose();
    }
  }

  try {
    const before = await predict();
    model.compile({ optimizer, loss: 'binaryCrossentropy' });
    await model.fit(pixels, pixels, { epochs: 2, shuffle: false, verbose: 0 });
    const after = await predict();
    assert.notDeepEqual(after, before);
  } finally {
    pixels.dispose();
    model.dispose();
    optimizer.dispose();
  }
  assert.equal(tf.memory().numTensors, baseline);
});
