import { createGeluLayer } from './gelu.js';
import { cosineLR } from './learning-rate.js';

/**
 * Autoencoder 784 → 128 → gargalo → 128 → 784.
 * Recebe e devolve arrays; todos os tensores e pesos pertencem à classe.
 * Não conhece DOM, Fashion-MNIST ou a projeção PCA.
 */
export class Autoencoder {
  constructor(tf, latentDimensions = 16) {
    this.tf = tf;
    this.latentDimensions = Math.max(
      2,
      Math.min(32, Math.floor(latentDimensions)),
    );
    this.training = false;
    this.stopRequested = false;
    const dense = (units, options = {}) =>
      tf.layers.dense({
        units,
        kernelInitializer: tf.initializers.glorotNormal({ seed: 42 }),
        ...options,
      });

    const input = tf.input({ shape: [784] });
    const hidden = createGeluLayer(tf).apply(dense(128).apply(input));
    const embedding = dense(this.latentDimensions, {
      name: 'bottleneck',
    }).apply(hidden);

    this.decoderLayers = [
      dense(128),
      createGeluLayer(tf),
      dense(784, { activation: 'sigmoid' }),
    ];
    const output = this.decoderLayers.reduce(
      (tensor, layer) => layer.apply(tensor),
      embedding,
    );
    this.model = tf.model({ inputs: input, outputs: output });
    this.encoder = tf.model({ inputs: input, outputs: embedding });
    this.optimizer = tf.train.adam(0.002);
    this.model.compile({
      optimizer: this.optimizer,
      loss: 'binaryCrossentropy',
    });
  }

  /** @param {Float32Array} pixels Imagem normalizada com 784 pixels. */
  async reconstruct(pixels) {
    const output = this.tf.tidy(() =>
      this.model.predict(this.tf.tensor2d(pixels, [1, 784])),
    );
    try {
      return await output.data();
    } finally {
      output.dispose();
    }
  }

  /** @returns {Promise<number[][]>} Um vetor do gargalo para cada imagem. */
  async encode(images) {
    const output = this.tf.tidy(() =>
      this.encoder.predict(this.tf.tensor2d(images, [images.length, 784])),
    );
    try {
      return await output.array();
    } finally {
      output.dispose();
    }
  }

  /** @returns {Promise<Float32Array>} Imagem gerada a partir de um vetor do gargalo. */
  async decode(embedding) {
    const output = this.tf.tidy(() =>
      this.decoderLayers.reduce(
        (tensor, layer) => layer.apply(tensor),
        this.tf.tensor2d(embedding, [1, this.latentDimensions]),
      ),
    );
    try {
      return await output.data();
    } finally {
      output.dispose();
    }
  }

  async train(
    images,
    { epochs, onEpochBegin = () => {}, onEpochEnd = () => {} },
  ) {
    if (this.training) throw new Error('O modelo já está treinando.');
    const inputs = this.tf.tensor2d(images, [images.length, 784]);
    this.training = true;
    this.stopRequested = false;
    this.model.stopTraining = false;
    try {
      await this.model.fit(inputs, inputs, {
        epochs,
        batchSize: Math.min(32, images.length),
        shuffle: false,
        callbacks: {
          onEpochBegin: async (epoch) => {
            const learningRate = cosineLR(epoch, epochs);
            this.optimizer.learningRate = learningRate;
            await onEpochBegin(epoch, learningRate);
          },
          onEpochEnd: async (epoch, logs) => {
            await onEpochEnd(epoch, Number(logs.loss));
            await this.tf.nextFrame();
            if (this.stopRequested) this.model.stopTraining = true;
          },
        },
      });
    } finally {
      this.training = false;
      inputs.dispose();
    }
  }

  stop() {
    this.stopRequested = true;
  }

  dispose() {
    if (this.training)
      throw new Error('Encerre o treino antes de descartar o modelo.');
    // Encoder e modelo completo compartilham as mesmas camadas. Liberá-las
    // pelo modelo completo evita descartar os pesos compartilhados duas vezes.
    this.model.dispose();
    this.optimizer.dispose();
  }
}
