/** Converte gestos de mouse/toque em zoom, deslocamento e seleção. */
export function bindCanvasNavigation(
  embeddingCanvas,
  { canInteract, onPan, onZoom, onTap },
) {
  const activePointers = new Map();
  let pointerGesture = null;
  function canvasPointFromClient(clientX, clientY) {
    const rect = embeddingCanvas.getBoundingClientRect();
    return {
      x: (clientX - rect.left) * (embeddingCanvas.width / rect.width),
      y: (clientY - rect.top) * (embeddingCanvas.height / rect.height),
    };
  }
  embeddingCanvas.addEventListener(
    'wheel',
    (event) => {
      if (!canInteract()) return;
      event.preventDefault();
      const p = canvasPointFromClient(event.clientX, event.clientY);
      onZoom(p.x, p.y, Math.exp(-event.deltaY * 0.0015));
    },
    { passive: false },
  );

  embeddingCanvas.addEventListener('pointerdown', (event) => {
    if (!canInteract()) return;
    embeddingCanvas.setPointerCapture(event.pointerId);
    const p = canvasPointFromClient(event.clientX, event.clientY);
    activePointers.set(event.pointerId, p);

    if (activePointers.size === 1) {
      pointerGesture = {
        type: 'pan',
        startX: p.x,
        startY: p.y,
        lastX: p.x,
        lastY: p.y,
        moved: false,
      };
    } else if (activePointers.size === 2) {
      const pts = [...activePointers.values()];
      const dx = pts[1].x - pts[0].x;
      const dy = pts[1].y - pts[0].y;
      pointerGesture = {
        type: 'pinch',
        lastDistance: Math.hypot(dx, dy),
        lastMidX: (pts[0].x + pts[1].x) / 2,
        lastMidY: (pts[0].y + pts[1].y) / 2,
      };
    }
  });

  embeddingCanvas.addEventListener('pointermove', (event) => {
    if (!activePointers.has(event.pointerId) || !canInteract()) return;

    const p = canvasPointFromClient(event.clientX, event.clientY);
    activePointers.set(event.pointerId, p);

    if (activePointers.size === 1 && pointerGesture?.type === 'pan') {
      const dx = p.x - pointerGesture.lastX;
      const dy = p.y - pointerGesture.lastY;

      if (
        Math.hypot(p.x - pointerGesture.startX, p.y - pointerGesture.startY) > 5
      ) {
        pointerGesture.moved = true;
      }

      if (pointerGesture.moved) {
        onPan(dx, dy);
      }

      pointerGesture.lastX = p.x;
      pointerGesture.lastY = p.y;
    } else if (activePointers.size === 2) {
      const pts = [...activePointers.values()];
      const dx = pts[1].x - pts[0].x;
      const dy = pts[1].y - pts[0].y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const midX = (pts[0].x + pts[1].x) / 2;
      const midY = (pts[0].y + pts[1].y) / 2;

      if (pointerGesture?.type !== 'pinch') {
        pointerGesture = {
          type: 'pinch',
          lastDistance: distance,
          lastMidX: midX,
          lastMidY: midY,
        };
        return;
      }

      onPan(midX - pointerGesture.lastMidX, midY - pointerGesture.lastMidY);

      onZoom(midX, midY, distance / pointerGesture.lastDistance);

      pointerGesture.lastDistance = distance;
      pointerGesture.lastMidX = midX;
      pointerGesture.lastMidY = midY;
    }
  });

  embeddingCanvas.addEventListener('pointerup', async (event) => {
    const wasTap =
      activePointers.size === 1 &&
      pointerGesture?.type === 'pan' &&
      !pointerGesture.moved;

    activePointers.delete(event.pointerId);

    if (embeddingCanvas.hasPointerCapture?.(event.pointerId)) {
      try {
        embeddingCanvas.releasePointerCapture(event.pointerId);
      } catch (_) {}
    }

    if (activePointers.size === 0) {
      if (wasTap) {
        await onTap(canvasPointFromClient(event.clientX, event.clientY));
      }
      pointerGesture = null;
    } else if (activePointers.size === 1) {
      const remaining = [...activePointers.values()][0];
      pointerGesture = {
        type: 'pan',
        startX: remaining.x,
        startY: remaining.y,
        lastX: remaining.x,
        lastY: remaining.y,
        moved: true,
      };
    }
  });

  embeddingCanvas.addEventListener('pointercancel', (event) => {
    activePointers.delete(event.pointerId);

    if (embeddingCanvas.hasPointerCapture?.(event.pointerId)) {
      try {
        embeddingCanvas.releasePointerCapture(event.pointerId);
      } catch (_) {}
    }

    if (activePointers.size === 0) pointerGesture = null;
  });

  embeddingCanvas.addEventListener('lostpointercapture', (event) => {
    activePointers.delete(event.pointerId);
    if (activePointers.size === 0) {
      pointerGesture = null;
    }
  });

  return {
    cancel() {
      for (const pointerId of activePointers.keys()) {
        if (embeddingCanvas.hasPointerCapture(pointerId)) {
          embeddingCanvas.releasePointerCapture(pointerId);
        }
      }
      activePointers.clear();
      pointerGesture = null;
    },
  };
}
