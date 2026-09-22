/**
 * Edite aqui a arquitetura do autoencoder usando a API Layers (estilo Keras).
 * Contrato: imagem [784] → embedding [embeddingSize] → imagem [784].
 * A saída sigmoid representa pixels normalizados entre 0 e 1.
 */
export function createAutoencoderModels(tf, embeddingSize) {
  // Encoder: transforma os pixels em uma representação compacta.
  const encoder = tf.sequential({
    name: 'encoder',
    layers: [
      tf.layers.inputLayer({ inputShape: [784] }),
      tf.layers.dense({
        units: 128,
        activation: 'relu',
      }),
      tf.layers.dense({
        units: embeddingSize,
        name: 'bottleneck',
      }),
    ],
  });

  // Decoder: reconstrói os pixels a partir da representação compacta.
  const decoder = tf.sequential({
    name: 'decoder',
    layers: [
      tf.layers.inputLayer({ inputShape: [embeddingSize] }),
      tf.layers.dense({
        units: 128,
        activation: 'relu',
      }),
      tf.layers.dense({
        units: 784,
        activation: 'sigmoid',
      }),
    ],
  });

  // Compõe as mesmas instâncias para que o treino atualize encoder e decoder.
  const input = tf.input({ shape: [784] });
  const embedding = encoder.apply(input);
  const reconstruction = decoder.apply(embedding);
  const model = tf.model({
    name: 'autoencoder',
    inputs: input,
    outputs: reconstruction,
  });

  return { encoder, decoder, model };
}
