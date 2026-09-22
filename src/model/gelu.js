/** A camada recebe o mesmo runtime TensorFlow usado pelo autoencoder. */
export function createGeluLayer(tf) {
  class GeluLayer extends tf.layers.Layer {
    computeOutputShape(inputShape) {
      return inputShape;
    }

    call(inputs) {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      return tf.tidy(() => x.mul(0.5).mul(tf.erf(x.div(Math.sqrt(2))).add(1)));
    }

    getClassName() {
      return 'GeluLayer';
    }
  }
  return new GeluLayer();
}
