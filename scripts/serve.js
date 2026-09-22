import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { extname, resolve, sep } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const port = Number(process.env.PORT || 8000);
const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.gz': 'application/gzip',
};

// Servidor local de desenvolvimento; a aplicação publicada só precisa de arquivos estáticos.
createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(
      new URL(request.url, 'http://localhost').pathname,
    );
    const relative = pathname === '/' ? 'index.html' : pathname.slice(1);
    const path = resolve(root, relative);
    const allowed =
      [
        'index.html',
        'autoencoder_tfjs_fashion_mnist_revista_limpo.html',
      ].includes(relative) ||
      relative.startsWith('src/') ||
      relative.startsWith('styles/') ||
      relative.startsWith('data/fashion-mnist/');
    if (!allowed || !path.startsWith(root.endsWith(sep) ? root : root + sep)) {
      response.writeHead(404).end('Not found');
      return;
    }
    const content = await readFile(path);
    response.writeHead(200, {
      'Content-Type': types[extname(path)] || 'application/octet-stream',
    });
    response.end(content);
  } catch (error) {
    response.writeHead(error.code === 'ENOENT' ? 404 : 400).end('Not found');
  }
}).listen(port, '127.0.0.1', () => {
  console.log(`Autoencoder disponível em http://127.0.0.1:${port}`);
});
