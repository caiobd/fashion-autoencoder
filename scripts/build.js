import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { build } from 'esbuild';
import * as datasetUrls from '../src/data/dataset-urls.js';
import { parseFashionMnist } from '../src/data/fashion-mnist.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const outputPath = resolve(root, 'dist/autoencoder.html');

async function embedDataset() {
  const files = await Promise.all(
    Object.entries(datasetUrls).map(async ([name, url]) => {
      let compressed;
      try {
        compressed = await readFile(new URL(url));
      } catch (error) {
        if (error.code === 'ENOENT') {
          throw new Error(
            'Dataset ausente. Execute git lfs pull antes de gerar o HTML.',
          );
        }
        throw error;
      }
      if (
        compressed
          .toString('utf8', 0, 80)
          .startsWith('version https://git-lfs.github.com/spec/v1')
      ) {
        throw new Error(
          'O dataset contém apenas ponteiros LFS. Execute git lfs pull antes de gerar o HTML.',
        );
      }
      const bytes = gunzipSync(compressed);
      return {
        name,
        dataUrl: `data:application/gzip;base64,${compressed.toString('base64')}`,
        buffer: bytes.buffer.slice(
          bytes.byteOffset,
          bytes.byteOffset + bytes.byteLength,
        ),
      };
    }),
  );
  const buffers = Object.fromEntries(
    files.map(({ name, buffer }) => [name, buffer]),
  );
  parseFashionMnist(buffers.FASHION_IMAGES_URL, buffers.FASHION_LABELS_URL);
  return files
    .map(
      ({ name, dataUrl }) =>
        `export const ${name} = ${JSON.stringify(dataUrl)};`,
    )
    .join('\n');
}

function replaceRequired(html, pattern, replacement) {
  if (!pattern.test(html))
    throw new Error(
      `Referência esperada não encontrada no index.html: ${pattern}`,
    );
  // A função de substituição preserva os caracteres $ presentes no JavaScript.
  return html.replace(pattern, () => replacement);
}

async function buildHtml() {
  const [template, css, datasetModule, tensorflowLicense, datasetLicense] =
    await Promise.all([
      readFile(resolve(root, 'index.html'), 'utf8'),
      readFile(resolve(root, 'styles/main.css'), 'utf8'),
      embedDataset(),
      readFile(resolve(root, 'licenses/tensorflowjs-LICENSE'), 'utf8'),
      readFile(resolve(root, 'data/fashion-mnist/LICENSE'), 'utf8'),
    ]);

  const bundle = await build({
    absWorkingDir: root,
    stdin: {
      contents: `import * as tf from '@tensorflow/tfjs';
globalThis.tf = tf;
await import('./src/main.js');`,
      resolveDir: root,
      sourcefile: 'standalone-entry.js',
    },
    bundle: true,
    platform: 'browser',
    format: 'esm',
    target: 'es2022',
    minify: true,
    write: false,
    legalComments: 'inline',
    // Evita que textos como </script> encerrem o elemento HTML prematuramente.
    supported: { 'inline-script': false },
    plugins: [
      {
        name: 'embed-fashion-mnist',
        setup(builder) {
          builder.onLoad({ filter: /dataset-urls\.js$/ }, ({ path }) => {
            if (path !== resolve(root, 'src/data/dataset-urls.js')) return;
            return { contents: datasetModule, loader: 'js' };
          });
        },
      },
    ],
  });

  let html = replaceRequired(
    template,
    /<script src="https:\/\/cdn\.jsdelivr\.net\/npm\/@tensorflow\/tfjs@[^"\s]+\/dist\/tf\.min\.js"><\/script>/,
    '',
  );
  html = replaceRequired(
    html,
    /<link rel="stylesheet" href="\.\/styles\/main\.css"\s*\/>/,
    `<style>\n${css}\n</style>`,
  );
  html = replaceRequired(
    html,
    /<script type="module" src="\.\/src\/main\.js"><\/script>/,
    `<script type="module">\n${bundle.outputFiles[0].text}\n</script>`,
  );

  const licenses = JSON.stringify({
    'TensorFlow.js (Apache-2.0)': tensorflowLicense,
    'Fashion-MNIST (MIT)': datasetLicense,
  }).replaceAll('<', '\\u003c');
  html = replaceRequired(
    html,
    /<\/body>/,
    `<script type="application/json" id="third-party-licenses">${licenses}</script>\n</body>`,
  );

  // Só grava o artefato depois de validar os dados e concluir o bundle.
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, html);
  console.log(
    `Gerado: ${outputPath} (${(Buffer.byteLength(html) / 1024 / 1024).toFixed(1)} MiB)`,
  );
}

try {
  await buildHtml();
} catch (error) {
  console.error('Não foi possível gerar o HTML:', error.message);
  process.exitCode = 1;
}
