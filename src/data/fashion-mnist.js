import { FASHION_IMAGES_URL, FASHION_LABELS_URL } from './dataset-urls.js';

export { FASHION_IMAGES_URL, FASHION_LABELS_URL } from './dataset-urls.js';

const IMAGE_SIZE = 28 * 28;
const CLASS_COUNT = 10;

/** Lê os dois arquivos IDX sem depender de TensorFlow ou da interface. */
export function parseFashionMnist(imageBuffer, labelBuffer) {
  if (imageBuffer.byteLength < 16 || labelBuffer.byteLength < 8) {
    throw new Error('Cabeçalho IDX incompleto.');
  }
  const images = new DataView(imageBuffer);
  const labels = new DataView(labelBuffer);
  const count = images.getUint32(4);
  if (images.getUint32(0) !== 2051 || labels.getUint32(0) !== 2049) {
    throw new Error('Formato IDX inválido.');
  }
  if (images.getUint32(8) !== 28 || images.getUint32(12) !== 28) {
    throw new Error('As imagens precisam ter 28 × 28 pixels.');
  }
  if (count === 0 || count !== labels.getUint32(4)) {
    throw new Error('Quantidade de imagens e labels inválida.');
  }
  if (
    imageBuffer.byteLength !== 16 + count * IMAGE_SIZE ||
    labelBuffer.byteLength !== 8 + count
  ) {
    throw new Error('Tamanho dos arquivos IDX não corresponde ao cabeçalho.');
  }
  const pixels = new Uint8Array(imageBuffer, 16);
  const classes = new Uint8Array(labelBuffer, 8);
  const byClass = Array.from({ length: CLASS_COUNT }, () => []);
  classes.forEach((label, index) => {
    if (label >= CLASS_COUNT) throw new Error(`Classe inválida: ${label}`);
    byClass[label].push(index);
  });
  return { pixels, labels: classes, byClass };
}

/** Mantém a mesma seleção determinística e alternada entre classes do HTML original. */
export function selectSamples(dataset, requestedCount) {
  const count = Math.floor(
    Math.max(
      10,
      Math.min(Number(requestedCount) || 300, dataset.labels.length),
    ),
  );
  if (dataset.byClass.some((indices) => indices.length === 0)) {
    throw new Error(
      'O conjunto precisa conter as dez classes do Fashion-MNIST.',
    );
  }
  const counters = new Uint32Array(CLASS_COUNT);
  const images = [];
  const labels = [];
  for (let index = 0; index < count; index++) {
    const label = index % CLASS_COUNT;
    const pool = dataset.byClass[label];
    const sourceIndex = pool[(counters[label]++ * 37) % pool.length];
    const start = sourceIndex * IMAGE_SIZE;
    images.push(
      Float32Array.from(
        dataset.pixels.subarray(start, start + IMAGE_SIZE),
        (pixel) => pixel / 255,
      ),
    );
    labels.push(label);
  }
  return { images, labels };
}

async function downloadGzip(url, description, onProgress) {
  onProgress(`Baixando ${description}…`);
  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok)
    throw new Error(`Falha ao baixar ${description}: HTTP ${response.status}`);
  if (typeof DecompressionStream === 'undefined') {
    throw new Error(
      'Este navegador não oferece DecompressionStream(gzip). Atualize o navegador.',
    );
  }
  const compressed = await response.arrayBuffer();
  // Um checkout sem os objetos LFS contém um pequeno arquivo de texto no lugar do gzip.
  const signature = new TextDecoder().decode(compressed.slice(0, 80));
  if (signature.startsWith('version https://git-lfs.github.com/spec/v1')) {
    throw new Error(
      'Os arquivos do dataset não foram baixados pelo Git LFS. Execute git lfs pull e recarregue a página.',
    );
  }
  onProgress(`Descompactando ${description}…`);
  const stream = new Blob([compressed])
    .stream()
    .pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

/** Carrega o split t10k original. O chamador mantém o resultado em memória. */
export async function loadFashionMnist(onProgress = () => {}) {
  // Aguarda ambos para que uma mensagem de progresso tardia não apague um erro.
  const downloads = await Promise.allSettled([
    downloadGzip(FASHION_IMAGES_URL, 'imagens Fashion-MNIST', onProgress),
    downloadGzip(FASHION_LABELS_URL, 'labels Fashion-MNIST', onProgress),
  ]);
  for (const download of downloads) {
    if (download.status === 'rejected') throw download.reason;
  }
  const [images, labels] = downloads.map((download) => download.value);
  const dataset = parseFashionMnist(images, labels);
  onProgress(
    `Fashion-MNIST pronto: ${dataset.labels.length.toLocaleString('pt-BR')} imagens reais 28×28.`,
  );
  return dataset;
}
