import { loadFashionMnist, selectSamples } from './data/fashion-mnist.js';
import { Autoencoder } from './model/autoencoder.js';
import { drawPixels } from './ui/canvas.js';
import { createEmbeddingExplorer } from './ui/embedding-explorer.js';

/** Orquestra o ciclo carregar → criar → treinar e atualiza os controles. */
export async function startApp(tf, elements) {
  const explorer = createEmbeddingExplorer(elements);
  const inputContext = elements.inputCanvas.getContext('2d');
  const outputContext = elements.outputCanvas.getContext('2d');
  let dataset = null;
  let autoencoder = null;
  let images = [];
  let currentSample = 0;
  let busy = false;
  let pendingPreview = Promise.resolve();

  const setStatus = (message) => {
    elements.status.textContent = message;
  };

  function setControls({ working, training = false }) {
    busy = working;
    for (const name of ['reset', 'latent', 'samples', 'epochs']) {
      elements[name].disabled = working;
    }
    elements.train.disabled = working || !autoencoder;
    elements.stop.disabled = !training;
    elements.next.disabled = !autoencoder || (working && !training);
  }

  function renderPrediction() {
    pendingPreview = pendingPreview
      .then(async () => {
        if (!autoencoder || !images.length) return;
        const pixels = images[currentSample % images.length];
        drawPixels(inputContext, pixels);
        drawPixels(outputContext, await autoencoder.reconstruct(pixels));
      })
      .catch((error) =>
        console.warn('Falha na prévia da reconstrução:', error),
      );
    return pendingPreview;
  }

  async function recreate() {
    if (busy) return;
    setControls({ working: true });
    setStatus('Recriando modelo…');
    try {
      await explorer.suspend();
      await pendingPreview;
      autoencoder?.dispose();
      autoencoder = null;
      dataset ??= await loadFashionMnist(setStatus);
      const samples = selectSamples(dataset, elements.samples.value);
      images = samples.images;
      const latent = Number(elements.latent.value) || 16;
      autoencoder = new Autoencoder(tf, latent);
      currentSample = 0;
      explorer.setModel(autoencoder, images, samples.labels);
      elements.epoch.textContent = '0';
      elements.loss.textContent = '—';
      elements.lr.textContent = '—';
      await renderPrediction();
      setStatus(`Modelo pronto: 784 → 128 → ${latent} → 128 → 784`);
      await explorer.update(false);
    } catch (error) {
      console.error(error);
      setStatus('Erro ao criar modelo: ' + error.message);
    } finally {
      setControls({ working: false });
    }
  }

  async function train() {
    if (busy || !autoencoder) return;
    setControls({ working: true, training: true });
    const epochs = Number(elements.epochs.value);
    try {
      await autoencoder.train(images, {
        epochs,
        onEpochBegin: (_epoch, rate) => {
          elements.lr.textContent = rate.toExponential(2);
        },
        onEpochEnd: async (epoch, loss) => {
          elements.epoch.textContent = `${epoch + 1}/${epochs}`;
          elements.loss.textContent = loss.toFixed(4);
          setStatus(`Treinando — época ${epoch + 1}/${epochs}`);
          if (epoch % 2 === 0 || epoch === epochs - 1) await renderPrediction();
          if (epoch % 5 === 0 || epoch === epochs - 1) await explorer.update();
        },
      });
      setStatus(
        autoencoder.stopRequested
          ? 'Treino interrompido.'
          : 'Treino concluído.',
      );
    } catch (error) {
      console.error(error);
      setStatus('Erro: ' + error.message);
    } finally {
      setControls({ working: false });
    }
  }

  elements.latent.addEventListener('input', () => {
    elements.latentValue.textContent = elements.latent.value;
    elements.latentLabel.textContent = elements.latent.value;
  });
  elements.latent.addEventListener('change', recreate);
  elements.samples.addEventListener('change', recreate);
  elements.reset.addEventListener('click', recreate);
  elements.train.addEventListener('click', train);
  elements.stop.addEventListener('click', () => {
    autoencoder?.stop();
    setStatus('Parando…');
  });
  elements.next.addEventListener('click', () => {
    currentSample = (currentSample + 1) % images.length;
    void renderPrediction();
  });

  setControls({ working: true });
  await tf.ready();
  setControls({ working: false });
  await recreate();
}
