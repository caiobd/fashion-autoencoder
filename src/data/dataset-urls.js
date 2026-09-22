// No HTML único, o build substitui este módulo pelos mesmos arquivos em data URLs.
export const FASHION_IMAGES_URL = new URL(
  '../../data/fashion-mnist/t10k-images-idx3-ubyte.gz',
  import.meta.url,
).href;
export const FASHION_LABELS_URL = new URL(
  '../../data/fashion-mnist/t10k-labels-idx1-ubyte.gz',
  import.meta.url,
).href;
