export function drawPixels(ctx, arr) {
  const img = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.max(0, Math.min(255, Math.round(arr[i] * 255)));
    const j = i * 4;
    img.data[j] = img.data[j + 1] = img.data[j + 2] = v;
    img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export function clearCanvas(context, canvas = context.canvas) {
  context.clearRect(0, 0, canvas.width, canvas.height);
}

export function makeThumbCanvas(pixels) {
  const canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;
  drawPixels(canvas.getContext('2d'), pixels);
  return canvas;
}
export function resizeCanvasBackingStore(canvas, maxDpr = 2) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return false;

  const dpr = Math.min(maxDpr, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));

  if (canvas.width === width && canvas.height === height) return false;

  canvas.width = width;
  canvas.height = height;
  return true;
}
