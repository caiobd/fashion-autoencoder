export function cosineLR(epoch, total) {
  const maxLR = 0.002;
  const minLR = 0.00005;
  const t = total <= 1 ? 1 : epoch / (total - 1);
  return minLR + 0.5 * (maxLR - minLR) * (1 + Math.cos(Math.PI * t));
}
