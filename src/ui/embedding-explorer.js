import { drawPixels, clearCanvas } from './canvas.js';
import { createEmbeddingPlot } from './embedding-plot.js';
import {
  pca2Model,
  procrustesAlignWithTransform,
  pcaPointToLatent,
} from '../math/projection.js';

/** Coordena seleção, projeção e interpolação sem conhecer camadas do modelo. */
export function createEmbeddingExplorer(elements) {
  const embeddingCanvas = elements.embeddingCanvas;
  const latentInputCanvas = elements.latentInputCanvas;
  const latentCanvas = elements.latentCanvas;
  const latentInputCaption = elements.latentInputCaption;
  const latentOutputCaption = elements.latentOutputCaption;
  const latentInfo = elements.latentInfo;
  const exploreModeBtn = elements.exploreModeBtn;
  const interpolateModeBtn = elements.interpolateModeBtn;
  const clearSelectionBtn = elements.clearSelectionBtn;
  const explorePanel = elements.explorePanel;
  const interpolatePanel = elements.interpolatePanel;
  const interpACanvas = elements.interpACanvas;
  const interpBCanvas = elements.interpBCanvas;
  const interpResultCanvas = elements.interpResultCanvas;
  const interpSlider = elements.interpSlider;
  const interpWeight = elements.interpWeight;
  const interpPlayBtn = elements.interpPlayBtn;
  const interpInfo = elements.interpInfo;
  const zoomInBtn = elements.zoomInBtn;
  const zoomOutBtn = elements.zoomOutBtn;
  const zoomResetBtn = elements.zoomResetBtn;
  const zoomReadout = elements.zoomReadout;
  const selectedInputContext = latentInputCanvas.getContext('2d');
  const selectedOutputContext = latentCanvas.getContext('2d');
  const interpolationAContext = interpACanvas.getContext('2d');
  const interpolationBContext = interpBCanvas.getContext('2d');
  const interpolationResultContext = interpResultCanvas.getContext('2d');

  let autoencoder = null;
  let rawData = [];
  let trackedIndices = [];
  let previousProjection = null;
  let currentPCA = null;
  let currentProcrustes = { c: 1, s: 0 };
  let currentTrackedLatents = null;
  let clickedAligned = null;
  let interactionMode = 'explore';
  let exploreSelectedIndex = null;
  let interpolationA = null;
  let interpolationB = null;
  let interpolationT = 0.5;
  let interpolationAnimating = false;
  let interpolationAnimationFrame = null;
  let interpolationDirection = 1;
  let interpolationLastTime = null;
  let pendingUpdate = Promise.resolve();
  let suspended = true;

  const plot = createEmbeddingPlot({
    canvas: embeddingCanvas,
    zoomReadout,
    zoomInBtn,
    zoomOutBtn,
    zoomResetBtn,
    onSelect: (index, point) => runSelection(() => explorePoint(index, point)),
  });

  // Uma leitura de tensor deve terminar antes de o app descartar o modelo.
  function enqueue(action) {
    pendingUpdate = pendingUpdate.then(action).catch((error) => {
      console.error(error);
      latentInfo.textContent = 'Erro na visualização: ' + error.message;
    });
    return pendingUpdate;
  }

  function runSelection(action) {
    if (suspended) return pendingUpdate;
    return enqueue(() => !suspended && action());
  }

  function drawEmbedding(points, animate = false) {
    plot.setSelection({
      interactionMode,
      exploreSelectedIndex,
      interpolationA,
      interpolationB,
      interpolationT,
      clickedAligned,
    });
    plot.render(points, animate);
  }

  function redrawEmbedding() {
    if (previousProjection) drawEmbedding(previousProjection);
  }
  async function showSampleFromEmbedding(sampleIndex) {
    if (!rawData[sampleIndex]) return;

    exploreSelectedIndex = sampleIndex;
    clickedAligned = null;

    const x = rawData[sampleIndex];
    drawPixels(selectedInputContext, x);

    latentInputCaption.textContent = 'Amostra selecionada';
    latentOutputCaption.textContent = 'Reconstrução da amostra';
    latentInfo.textContent = 'Reconstruindo a miniatura selecionada…';

    const decoded = await autoencoder.reconstruct(x);
    if (decoded) {
      drawPixels(selectedOutputContext, decoded);
      latentInfo.innerHTML =
        '<b>Amostra real do Fashion-MNIST</b><br>' +
        'Resolução: 28 × 28 pixels.';
    }

    redrawEmbedding();
  }

  function trackedPositionForSample(sampleIndex) {
    const i = trackedIndices.indexOf(sampleIndex);
    return i >= 0 ? i : null;
  }

  function latentForTrackedSample(sampleIndex) {
    const i = trackedPositionForSample(sampleIndex);
    if (i === null || !currentTrackedLatents?.[i]) return null;
    return currentTrackedLatents[i];
  }

  function setInteractionMode(mode) {
    if (mode !== 'interpolate') {
      stopInterpolationAnimation();
    }

    interactionMode = mode;
    const exploring = mode === 'explore';

    plot.cancelGesture();

    exploreModeBtn.classList.toggle('active', exploring);
    interpolateModeBtn.classList.toggle('active', !exploring);
    explorePanel.classList.toggle('hidden', !exploring);
    interpolatePanel.classList.toggle('hidden', exploring);

    clickedAligned = null;
    redrawEmbedding();
  }

  function clearBottomSelection() {
    stopInterpolationAnimation();
    exploreSelectedIndex = null;
    clickedAligned = null;
    interpolationA = null;
    interpolationB = null;
    interpolationT = 0.5;

    interpSlider.value = '50';
    interpSlider.disabled = true;
    interpPlayBtn.disabled = true;
    interpWeight.textContent = 'Selecione duas miniaturas';
    interpInfo.textContent =
      'No modo de interpolação, toque em uma miniatura para A e depois em outra para B.';

    clearCanvas(selectedInputContext, latentInputCanvas);
    clearCanvas(selectedOutputContext, latentCanvas);
    clearCanvas(interpolationAContext, interpACanvas);
    clearCanvas(interpolationBContext, interpBCanvas);
    clearCanvas(interpolationResultContext, interpResultCanvas);

    latentInputCaption.textContent = 'Amostra selecionada';
    latentOutputCaption.textContent = 'Reconstrução / ponto PCA';
    latentInfo.textContent =
      'Toque numa miniatura para ampliá-la e ver sua reconstrução. Toque no espaço vazio para decodificar aquele ponto usando o decoder.';

    redrawEmbedding();
  }

  function stopInterpolationAnimation() {
    interpolationAnimating = false;
    interpolationLastTime = null;

    if (interpolationAnimationFrame !== null) {
      cancelAnimationFrame(interpolationAnimationFrame);
      interpolationAnimationFrame = null;
    }

    interpPlayBtn.textContent = 'Animar A ↔ B';
  }

  function startInterpolationAnimation() {
    if (interpolationA === null || interpolationB === null) return;

    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      interpInfo.textContent =
        'A animação automática está desativada pela preferência de movimento reduzido do sistema.';
      return;
    }

    interpolationAnimating = true;
    interpolationLastTime = null;
    interpPlayBtn.textContent = 'Pausar animação';

    // Uma travessia completa A -> B leva ~3 s.
    const speed = 1 / 3000;

    const tick = async (now) => {
      if (!interpolationAnimating) return;

      if (interpolationLastTime === null) {
        interpolationLastTime = now;
      }

      const dt = Math.min(50, now - interpolationLastTime);
      interpolationLastTime = now;

      let next = interpolationT + interpolationDirection * dt * speed;

      if (next >= 1) {
        next = 1;
        interpolationDirection = -1;
      } else if (next <= 0) {
        next = 0;
        interpolationDirection = 1;
      }

      interpolationT = next;
      interpSlider.value = String(Math.round(next * 100));
      await enqueue(
        () => interpolationAnimating && updateInterpolationResult(),
      );

      if (interpolationAnimating) {
        interpolationAnimationFrame = requestAnimationFrame(tick);
      }
    };

    interpolationAnimationFrame = requestAnimationFrame(tick);
  }

  function toggleInterpolationAnimation() {
    if (interpolationAnimating) {
      stopInterpolationAnimation();
    } else {
      startInterpolationAnimation();
    }
  }

  async function chooseInterpolationSample(sampleIndex) {
    if (!rawData[sampleIndex]) return;

    stopInterpolationAnimation();

    if (
      interpolationA === null ||
      (interpolationA !== null && interpolationB !== null)
    ) {
      interpolationA = sampleIndex;
      interpolationB = null;
      interpolationT = 0.5;
      interpSlider.value = '50';
      interpSlider.disabled = true;
      interpPlayBtn.disabled = true;

      drawPixels(interpolationAContext, rawData[sampleIndex]);
      clearCanvas(interpolationBContext, interpBCanvas);
      clearCanvas(interpolationResultContext, interpResultCanvas);

      interpWeight.textContent = 'A selecionada — escolha B';
      interpInfo.textContent =
        'Agora toque em uma segunda miniatura para definir B.';
    } else {
      interpolationB = sampleIndex;
      drawPixels(interpolationBContext, rawData[sampleIndex]);
      interpSlider.disabled = false;
      interpPlayBtn.disabled = false;
      interpInfo.textContent =
        'Arraste o slider para percorrer o segmento A ↔ B no embedding.';
      await updateInterpolationResult();
    }

    redrawEmbedding();
  }

  async function updateInterpolationResult() {
    if (interpolationA === null || interpolationB === null) return;

    const zA = latentForTrackedSample(interpolationA);
    const zB = latentForTrackedSample(interpolationB);
    if (!zA || !zB) return;

    const t = interpolationT;
    const z = new Float32Array(zA.length);

    for (let j = 0; j < z.length; j++) {
      z[j] = (1 - t) * zA[j] + t * zB[j];
    }

    const decoded = await autoencoder.decode(z);
    if (decoded) drawPixels(interpolationResultContext, decoded);

    interpWeight.textContent = `A ${Math.round((1 - t) * 100)}%  •  B ${Math.round(t * 100)}%`;

    redrawEmbedding();
  }

  async function updateProjection(animate) {
    if (!autoencoder || !trackedIndices.length) return;
    currentTrackedLatents = await autoencoder.encode(
      trackedIndices.map((index) => rawData[index]),
    );
    currentPCA = pca2Model(currentTrackedLatents);
    const aligned = procrustesAlignWithTransform(
      currentPCA.points,
      previousProjection,
    );
    currentProcrustes = { c: aligned.c, s: aligned.s };
    previousProjection = aligned.points;
    drawEmbedding(previousProjection, animate);
    if (
      interactionMode === 'interpolate' &&
      interpolationA !== null &&
      interpolationB !== null
    ) {
      await updateInterpolationResult();
    }
  }

  async function explorePoint(index, aligned) {
    if (!autoencoder || !currentPCA) return;
    if (index !== null) {
      if (interactionMode === 'interpolate')
        await chooseInterpolationSample(index);
      else await showSampleFromEmbedding(index);
      return;
    }
    if (interactionMode === 'interpolate') {
      interpInfo.textContent =
        'No modo de interpolação, selecione as duas extremidades tocando nas miniaturas.';
      return;
    }
    clickedAligned = aligned;
    clearCanvas(selectedInputContext);
    latentInputCaption.textContent = 'Interpolação (sem amostra direta)';
    latentOutputCaption.textContent = 'Decodificação do ponto PCA';
    latentInfo.textContent = 'Decodificando ponto escolhido…';
    const latent = pcaPointToLatent(aligned, currentPCA, currentProcrustes);
    drawPixels(selectedOutputContext, await autoencoder.decode(latent));
    latentInfo.innerHTML =
      '<b>Saída do decoder para o ponto selecionado</b><br>' +
      'Esta imagem foi gerada a partir de um ponto do espaço de embeddings que não corresponde diretamente a nenhuma imagem do nosso conjunto de dados. ' +
      'O decoder está mostrando como seria uma imagem descrita por esse ponto.';
    redrawEmbedding();
  }

  exploreModeBtn.addEventListener('click', () =>
    runSelection(() => setInteractionMode('explore')),
  );
  interpolateModeBtn.addEventListener('click', () =>
    runSelection(() => setInteractionMode('interpolate')),
  );
  clearSelectionBtn.addEventListener('click', () =>
    runSelection(clearBottomSelection),
  );
  interpSlider.addEventListener('pointerdown', (event) => {
    plot.cancelGesture();
    event.stopPropagation();
  });
  interpSlider.addEventListener('input', () => {
    stopInterpolationAnimation();
    const value = Number(interpSlider.value) / 100;
    runSelection(async () => {
      interpolationT = value;
      await updateInterpolationResult();
    });
  });
  interpPlayBtn.addEventListener('click', () =>
    runSelection(toggleInterpolationAnimation),
  );
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopInterpolationAnimation();
  });
  setInteractionMode('explore');

  return {
    async suspend() {
      suspended = true;
      stopInterpolationAnimation();
      await pendingUpdate;
      autoencoder = null;
    },
    setModel(model, images, labels) {
      suspended = false;
      autoencoder = model;
      rawData = images;
      trackedIndices = Array.from(
        { length: Math.min(images.length, 80) },
        (_, index) => index,
      );
      previousProjection = null;
      currentPCA = null;
      currentTrackedLatents = null;
      plot.setSamples(images, labels, trackedIndices);
      clearBottomSelection();
    },
    update(animate = true) {
      return enqueue(async () => {
        try {
          await updateProjection(animate);
        } catch (error) {
          console.warn('Falha na projeção dos embeddings:', error);
          plot.showError();
        }
      });
    },
  };
}
