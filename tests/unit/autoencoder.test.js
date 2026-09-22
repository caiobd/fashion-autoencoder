import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import { Autoencoder } from '../../src/model/autoencoder.js';
import { cosineLR } from '../../src/model/learning-rate.js';

await tf.setBackend('cpu');
await tf.ready();
const pixels = Float32Array.from(
  { length: 784 },
  (_, index) => (index % 28) / 27,
);

test('encoder e decoder compartilham os pesos da reconstrução completa', async () => {
  for (const dimensions of [2, 16, 32]) {
    const model = new Autoencoder(tf, dimensions);
    try {
      const [embedding] = await model.encode([pixels]);
      assert.equal(embedding.length, dimensions);
      const decoded = await model.decode(embedding);
      const reconstructed = await model.reconstruct(pixels);
      assert.equal(reconstructed.length, 784);
      reconstructed.forEach((value, index) => {
        assert.ok(value >= 0 && value <= 1);
        assert.ok(Math.abs(value - decoded[index]) < 1e-6);
      });
    } finally {
      model.dispose();
    }
  }
});

test('o treino altera a reconstrução, permite parar e pode ser retomado', async () => {
  const model = new Autoencoder(tf, 2);
  try {
    const before = await model.reconstruct(pixels);
    const epochs = [];
    await model.train([pixels], {
      epochs: 5,
      onEpochEnd: (epoch, loss) => {
        assert.ok(Number.isFinite(loss));
        epochs.push(epoch);
        model.stop();
      },
    });
    assert.deepEqual(epochs, [0]);
    assert.notDeepEqual(await model.reconstruct(pixels), before);
    const resumed = [];
    await model.train([pixels], {
      epochs: 2,
      onEpochEnd: (epoch) => resumed.push(epoch),
    });
    assert.deepEqual(resumed, [0, 1]);
  } finally {
    model.dispose();
  }
});

test('libera tensores após inferência, treino e falhas em callbacks', async () => {
  const baseline = tf.memory().numTensors;
  for (let cycle = 0; cycle < 3; cycle++) {
    const model = new Autoencoder(tf, 2);
    await model.reconstruct(pixels);
    await model.encode([pixels]);
    await model.decode([0, 0]);
    await model.train([pixels], { epochs: 1 });
    await assert.rejects(
      model.train([pixels], {
        epochs: 1,
        onEpochEnd: () => {
          throw new Error('Falha simulada');
        },
      }),
      /Falha simulada/,
    );
    model.dispose();
    assert.equal(tf.memory().numTensors, baseline);
  }
});

test('mantém os limites e o decaimento cosseno da taxa de aprendizado', () => {
  assert.equal(cosineLR(0, 100), 0.002);
  assert.equal(cosineLR(99, 100), 0.00005);
  assert.equal(cosineLR(0, 1), 0.00005);
  assert.ok(cosineLR(49, 100) > cosineLR(50, 100));
});
