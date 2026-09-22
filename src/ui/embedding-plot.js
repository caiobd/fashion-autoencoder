import { makeThumbCanvas, resizeCanvasBackingStore } from './canvas.js';
import { bindCanvasNavigation } from './canvas-navigation.js';

/** Desenha a projeção e cuida apenas da navegação no canvas. */
export function createEmbeddingPlot({
  canvas: embeddingCanvas,
  zoomReadout,
  zoomInBtn,
  zoomOutBtn,
  zoomResetBtn,
  onSelect,
}) {
  const context = embeddingCanvas.getContext('2d');
  let trackedIndices = [];
  let rawLabels = [];
  let trackedThumbs = [];
  let previousProjection = null;
  let previousDrawProjection = null;
  let embeddingView = null;
  let embeddingHitboxes = [];
  let animationFrame = null;
  let interactionMode = 'explore';
  let exploreSelectedIndex = null;
  let interpolationA = null;
  let interpolationB = null;
  let interpolationT = 0.5;
  let clickedAligned = null;
  const viewState = { zoom: 1, panX: 0, panY: 0, minZoom: 0.5, maxZoom: 12 };
  function trackedPositionForSample(sampleIndex) {
    const i = trackedIndices.indexOf(sampleIndex);
    return i >= 0 ? i : null;
  }

  function projectionForSample(sampleIndex) {
    const i = trackedPositionForSample(sampleIndex);
    if (i === null || !previousProjection?.[i]) return null;
    return previousProjection[i];
  }

  function updateZoomReadout() {
    zoomReadout.textContent = Math.round(viewState.zoom * 100) + '%';
  }

  function screenFromEmbedding(point) {
    if (!embeddingView) return [0, 0];
    return [
      embeddingView.cx +
        viewState.panX +
        point[0] * embeddingView.baseScale * viewState.zoom,
      embeddingView.cy +
        viewState.panY -
        point[1] * embeddingView.baseScale * viewState.zoom,
    ];
  }

  function embeddingFromScreen(x, y) {
    if (!embeddingView) return [0, 0];
    const scale = embeddingView.baseScale * viewState.zoom;
    return [
      (x - embeddingView.cx - viewState.panX) / scale,
      (embeddingView.cy + viewState.panY - y) / scale,
    ];
  }

  function drawEmbedding(points, animate = true) {
    cancelAnimationFrame(animationFrame);
    previousProjection = points;
    const w = embeddingCanvas.width,
      h = embeddingCanvas.height;
    context.fillStyle = '#fbfaf4';
    context.fillRect(0, 0, w, h);

    if (!points || points.length === 0) return;

    let maxAbs = 1e-6;
    for (const [x, y] of points)
      maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
    const baseScale = (0.42 * Math.min(w, h)) / maxAbs;

    embeddingView = { cx: w / 2, cy: h / 2, baseScale };

    const target = points.map((p, i) => {
      const [x, y] = screenFromEmbedding(p);
      return {
        x,
        y,
        cls: rawLabels[trackedIndices[i]] ?? 0,
        idx: trackedIndices[i],
      };
    });

    const prev =
      previousDrawProjection && previousDrawProjection.length === target.length
        ? previousDrawProjection
        : target;

    const palette = [
      '#7aa2ff',
      '#85e0b4',
      '#f1c75b',
      '#e88973',
      '#b89cff',
      '#70d6e8',
      '#e89cc8',
      '#c5db75',
      '#f0a35e',
      '#9db7d5',
    ];

    const baseThumb = target.length <= 30 ? 26 : target.length <= 60 ? 21 : 17;
    const thumbSize = Math.max(
      12,
      Math.min(34, baseThumb * Math.sqrt(viewState.zoom)),
    );
    const half = thumbSize / 2;

    const drawFrame = (t) => {
      embeddingHitboxes = [];
      context.fillStyle = '#fbfaf4';
      context.fillRect(0, 0, w, h);

      const centerX = embeddingView.cx + viewState.panX;
      const centerY = embeddingView.cy + viewState.panY;
      context.strokeStyle = '#c4d0c7';
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(centerX, 0);
      context.lineTo(centerX, h);
      context.moveTo(0, centerY);
      context.lineTo(w, centerY);
      context.stroke();

      for (let i = 0; i < target.length; i++) {
        const x = prev[i].x + (target[i].x - prev[i].x) * t;
        const y = prev[i].y + (target[i].y - prev[i].y) * t;
        const thumb = trackedThumbs[i];

        if (
          x < -thumbSize ||
          x > w + thumbSize ||
          y < -thumbSize ||
          y > h + thumbSize
        )
          continue;

        if (thumb) {
          context.strokeStyle = palette[target[i].cls];
          context.lineWidth = 2;
          context.strokeRect(
            x - half - 1,
            y - half - 1,
            thumbSize + 2,
            thumbSize + 2,
          );

          context.imageSmoothingEnabled = false;
          context.drawImage(thumb, x - half, y - half, thumbSize, thumbSize);

          embeddingHitboxes.push({
            left: x - half - 4,
            right: x + half + 4,
            top: y - half - 4,
            bottom: y + half + 4,
            idx: target[i].idx,
            projectionIndex: i,
          });

          if (
            interactionMode === 'explore' &&
            target[i].idx === exploreSelectedIndex
          ) {
            context.strokeStyle = '#0d513d';
            context.lineWidth = 2.5;
            context.strokeRect(
              x - half - 4,
              y - half - 4,
              thumbSize + 8,
              thumbSize + 8,
            );
          }
        }
      }

      if (interactionMode === 'interpolate' && interpolationA !== null) {
        const pA = projectionForSample(interpolationA);
        const pB =
          interpolationB !== null ? projectionForSample(interpolationB) : null;

        if (pA) {
          const [ax, ay] = screenFromEmbedding(pA);

          context.beginPath();
          context.strokeStyle = '#0d513d';
          context.lineWidth = 3;
          context.arc(ax, ay, 11, 0, Math.PI * 2);
          context.stroke();
          context.fillStyle = '#fbfaf4';
          context.font = 'bold 11px system-ui';
          context.textAlign = 'center';
          context.textBaseline = 'middle';
          context.fillText('A', ax, ay);

          if (pB) {
            const [bx, by] = screenFromEmbedding(pB);

            context.beginPath();
            context.strokeStyle = '#0d513d';
            context.lineWidth = 3;
            context.arc(bx, by, 11, 0, Math.PI * 2);
            context.stroke();
            context.fillStyle = '#fbfaf4';
            context.fillText('B', bx, by);

            // Segmento percorrido pela interpolação.
            context.beginPath();
            context.strokeStyle = 'rgba(13,81,61,.62)';
            context.lineWidth = 2.5;
            context.moveTo(ax, ay);
            context.lineTo(bx, by);
            context.stroke();

            const p = [
              (1 - interpolationT) * pA[0] + interpolationT * pB[0],
              (1 - interpolationT) * pA[1] + interpolationT * pB[1],
            ];
            const [ix, iy] = screenFromEmbedding(p);

            // Ponto atual da interpolação.
            context.beginPath();
            context.fillStyle = '#0d513d';
            context.arc(ix, iy, 6, 0, Math.PI * 2);
            context.fill();

            context.beginPath();
            context.strokeStyle = '#fbfaf4';
            context.lineWidth = 2;
            context.arc(ix, iy, 9, 0, Math.PI * 2);
            context.stroke();
          }
        }
      }

      if (clickedAligned && embeddingView) {
        const [px, py] = screenFromEmbedding(clickedAligned);
        context.beginPath();
        context.strokeStyle = '#0d513d';
        context.lineWidth = 3;
        context.arc(px, py, 10, 0, Math.PI * 2);
        context.stroke();
        context.beginPath();
        context.moveTo(px - 14, py);
        context.lineTo(px + 14, py);
        context.moveTo(px, py - 14);
        context.lineTo(px, py + 14);
        context.stroke();
      }
    };

    updateZoomReadout();

    if (!animate || matchMedia('(prefers-reduced-motion: reduce)').matches) {
      drawFrame(1);
      previousDrawProjection = target;
      return;
    }

    const start = performance.now();
    const duration = 280;

    function step(now) {
      const u = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - u, 3);
      drawFrame(eased);
      if (u < 1) animationFrame = requestAnimationFrame(step);
      else previousDrawProjection = target;
    }

    animationFrame = requestAnimationFrame(step);
  }

  function redrawEmbedding() {
    if (!previousProjection) return;
    previousDrawProjection = null;
    drawEmbedding(previousProjection, false);
  }

  function zoomAt(canvasX, canvasY, factor) {
    if (!embeddingView || !previousProjection) return;

    const before = embeddingFromScreen(canvasX, canvasY);
    const nextZoom = Math.max(
      viewState.minZoom,
      Math.min(viewState.maxZoom, viewState.zoom * factor),
    );

    if (nextZoom === viewState.zoom) return;
    viewState.zoom = nextZoom;

    const scale = embeddingView.baseScale * viewState.zoom;
    viewState.panX = canvasX - embeddingView.cx - before[0] * scale;
    viewState.panY = canvasY - embeddingView.cy + before[1] * scale;

    redrawEmbedding();
  }

  function resetEmbeddingView() {
    viewState.zoom = 1;
    viewState.panX = 0;
    viewState.panY = 0;
    redrawEmbedding();
  }

  function hitTestEmbeddingThumbnail(canvasX, canvasY) {
    // Recorre de trás para frente para selecionar a miniatura desenhada por cima
    // quando há sobreposição.
    for (let i = embeddingHitboxes.length - 1; i >= 0; i--) {
      const b = embeddingHitboxes[i];
      if (
        canvasX >= b.left &&
        canvasX <= b.right &&
        canvasY >= b.top &&
        canvasY <= b.bottom
      ) {
        return b;
      }
    }
    return null;
  }

  const navigation = bindCanvasNavigation(embeddingCanvas, {
    canInteract: () => Boolean(previousProjection),
    onPan: (x, y) => {
      viewState.panX += x;
      viewState.panY += y;
      redrawEmbedding();
    },
    onZoom: zoomAt,
    onTap: ({ x, y }) =>
      onSelect(
        hitTestEmbeddingThumbnail(x, y)?.idx ?? null,
        embeddingFromScreen(x, y),
      ),
  });
  zoomInBtn.addEventListener('click', () =>
    zoomAt(embeddingCanvas.width / 2, embeddingCanvas.height / 2, 1.35),
  );
  zoomOutBtn.addEventListener('click', () =>
    zoomAt(embeddingCanvas.width / 2, embeddingCanvas.height / 2, 1 / 1.35),
  );
  zoomResetBtn.addEventListener('click', resetEmbeddingView);
  const resize = () => {
    if (resizeCanvasBackingStore(embeddingCanvas)) redrawEmbedding();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(embeddingCanvas);
  window.addEventListener('orientationchange', () => setTimeout(resize, 50));
  resize();

  return {
    render: drawEmbedding,
    cancelGesture: () => navigation.cancel(),
    setSelection(selection) {
      ({
        interactionMode,
        exploreSelectedIndex,
        interpolationA,
        interpolationB,
        interpolationT,
        clickedAligned,
      } = selection);
    },
    setSamples(images, labels, indices) {
      cancelAnimationFrame(animationFrame);
      navigation.cancel();
      rawLabels = labels;
      trackedIndices = indices;
      trackedThumbs = indices.map((index) => makeThumbCanvas(images[index]));
      previousProjection = null;
      previousDrawProjection = null;
      embeddingView = null;
      embeddingHitboxes = [];
      resetEmbeddingView();
      updateZoomReadout();
      context.clearRect(0, 0, embeddingCanvas.width, embeddingCanvas.height);
    },
    showError() {
      cancelAnimationFrame(animationFrame);
      const { width, height } = embeddingCanvas;
      context.fillStyle = '#fbfaf4';
      context.fillRect(0, 0, width, height);
      context.fillStyle = '#60736b';
      context.font = '14px system-ui';
      context.textAlign = 'center';
      context.fillText(
        'Projeção indisponível nesta atualização',
        width / 2,
        height / 2,
      );
    },
  };
}
