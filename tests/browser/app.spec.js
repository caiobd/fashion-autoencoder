import { test, expect } from '@playwright/test';

async function configureNetwork(page) {
  // O dataset é servido pelo app. Apenas o TF.js do CDN é substituído pela cópia local.
  const externalRequests = [];
  const localOrigin = new URL(test.info().project.use.baseURL).origin;
  await page.route('**/*', (route) => {
    if (new URL(route.request().url()).origin === localOrigin) {
      return route.continue();
    }
    externalRequests.push(route.request().url());
    return route.abort();
  });
  await page.route(
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
    (route) =>
      route.fulfill({
        path: 'node_modules/@tensorflow/tfjs/dist/tf.min.js',
        contentType: 'text/javascript',
      }),
  );
  return externalRequests;
}

async function openApp(page) {
  const externalRequests = await configureNetwork(page);
  await page.goto('/');
  await expect(page.locator('#train')).toBeEnabled();
  await expect(page.locator('#status')).toContainText('Modelo pronto');
  expect(externalRequests).toEqual([]);
}

test('orienta a obter os objetos quando o checkout contém apenas ponteiros LFS', async ({
  page,
}) => {
  await configureNetwork(page);
  await page.route('**/data/fashion-mnist/t10k-images-idx3-ubyte.gz', (route) =>
    route.fulfill({
      body: 'version https://git-lfs.github.com/spec/v1\noid sha256:346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073\nsize 4422102\n',
      contentType: 'application/gzip',
    }),
  );
  await page.goto('/');
  await expect(page.locator('#status')).toContainText('Execute git lfs pull');
  await expect(page.locator('#train')).toBeDisabled();
});

async function thumbnailPositions(page) {
  // Localiza duas bordas coloridas no desenho real, sem depender da inicialização dos pesos.
  return page.evaluate(() => {
    const canvas = document.getElementById('embeddingCanvas');
    const rect = canvas.getBoundingClientRect();
    const pixels = canvas
      .getContext('2d')
      .getImageData(0, 0, canvas.width, canvas.height).data;
    let first = null;
    let farthest = null;
    let maxDistance = 0;
    for (let offset = 0; offset < pixels.length; offset += 4) {
      const [red, green, blue] = pixels.subarray(offset, offset + 3);
      // Imagens são cinza; fundo e eixos têm pouca diferença entre os canais.
      if (Math.max(red, green, blue) - Math.min(red, green, blue) <= 50)
        continue;
      const x = (offset / 4) % canvas.width;
      const y = Math.floor(offset / 4 / canvas.width);
      first ??= { x, y };
      const distance = (x - first.x) ** 2 + (y - first.y) ** 2;
      if (distance > maxDistance) {
        maxDistance = distance;
        farthest = { x, y };
      }
    }
    if (!first || !farthest || maxDistance < 40 ** 2) {
      throw new Error(
        'Não foram encontradas duas miniaturas separadas no canvas.',
      );
    }
    return [first, farthest].map(({ x, y }) => ({
      x: ((x + 0.5) * rect.width) / canvas.width,
      y: ((y + 0.5) * rect.height) / canvas.height,
    }));
  });
}

test('carrega os dados reais, explora, interpola, treina, para e recria', async ({
  page,
}) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await openApp(page);
  await page.locator('#samples').selectOption('10');
  await expect(page.locator('#train')).toBeEnabled();
  const original = await page
    .locator('#inputCanvas')
    .evaluate((canvas) => canvas.toDataURL());
  await page.locator('#next').click();
  await expect
    .poll(() =>
      page.locator('#inputCanvas').evaluate((canvas) => canvas.toDataURL()),
    )
    .not.toBe(original);

  await page.locator('#zoomIn').click();
  await expect(page.locator('#zoomReadout')).toHaveText('135%');
  await page.locator('#zoomReset').click();
  await expect(page.locator('#zoomReadout')).toHaveText('100%');

  const canvas = page.locator('#embeddingCanvas');
  await canvas.scrollIntoViewIfNeeded();
  const bounds = await canvas.boundingBox();
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + 100);
  await page.mouse.wheel(0, -200);
  await expect(page.locator('#zoomReadout')).not.toHaveText('100%');
  await page.locator('#zoomReset').click();
  const beforePan = await canvas.evaluate((element) => element.toDataURL());
  await canvas.scrollIntoViewIfNeeded();
  const panBounds = await canvas.boundingBox();
  await page.mouse.move(panBounds.x + 50, panBounds.y + 50);
  await page.mouse.down();
  await page.mouse.move(panBounds.x + 90, panBounds.y + 90, { steps: 5 });
  await page.mouse.up();
  await expect
    .poll(() => canvas.evaluate((element) => element.toDataURL()))
    .not.toBe(beforePan);
  await page.locator('#zoomReset').click();

  const points = await thumbnailPositions(page);
  await page.locator('#embeddingCanvas').click({ position: points[0] });
  await expect(page.locator('#latentInfo')).toContainText('Amostra real');
  await page.locator('#embeddingCanvas').click({ position: { x: 5, y: 5 } });
  await expect(page.locator('#latentInfo')).toContainText('Saída do decoder');
  await page.locator('#interpolateModeBtn').click();
  await page.locator('#embeddingCanvas').click({ position: points[0] });
  await page.locator('#embeddingCanvas').click({ position: points[1] });
  await expect(page.locator('#interpSlider')).toBeEnabled();
  await expect(page.locator('#interpWeight')).toHaveText('A 50%  •  B 50%');
  await page.locator('#interpSlider').fill('25');
  await expect(page.locator('#interpWeight')).toHaveText('A 75%  •  B 25%');
  await page.locator('#interpPlayBtn').click();
  await expect(page.locator('#interpInfo')).toContainText('movimento reduzido');
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.locator('#interpPlayBtn').click();
  await expect(page.locator('#interpPlayBtn')).toHaveText('Pausar animação');
  await expect(page.locator('#interpSlider')).not.toHaveValue('25');
  await page.locator('#interpPlayBtn').click();
  await expect(page.locator('#interpPlayBtn')).toHaveText('Animar A ↔ B');

  await page.locator('#epochs').selectOption('100');
  await page.locator('#train').click();
  await expect(page.locator('#loss')).not.toHaveText('—');
  await page.locator('#stop').click();
  await expect(page.locator('#status')).toHaveText('Treino interrompido.');
  await expect(page.locator('#train')).toBeEnabled();
  await page.locator('#reset').click();
  await expect(page.locator('#train')).toBeEnabled();
  await expect(page.locator('#epoch')).toHaveText('0');
  await expect(page.locator('#interpSlider')).toBeDisabled();
  await page.locator('#train').click();
  await expect(page.locator('#status')).toHaveText('Treino concluído.');
  await expect(page.locator('#epoch')).toHaveText('100/100');
  await page.locator('#latent').fill('2');
  await page.locator('#latent').dispatchEvent('change');
  await expect(page.locator('#train')).toBeEnabled();
  await expect(page.locator('#status')).toHaveText(
    'Modelo pronto: 784 → 128 → 2 → 128 → 784',
  );
  const tensorCount = await page.evaluate(() => window.tf.memory().numTensors);
  for (let cycle = 0; cycle < 3; cycle++) {
    await page.locator('#reset').click();
    await expect(page.locator('#train')).toBeEnabled();
    expect(await page.evaluate(() => window.tf.memory().numTensors)).toBe(
      tensorCount,
    );
  }
  expect(errors).toEqual([]);
});

test('mantém a página utilizável em telas pequenas e no endereço antigo', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openApp(page);
  await expect(page.locator('#embeddingCanvas')).toBeVisible();
  await page.locator('#zoomIn').click();
  await expect(page.locator('#zoomReadout')).toHaveText('135%');
  await page.goto('/autoencoder_tfjs_fashion_mnist_revista_limpo.html');
  await expect(page).toHaveURL(/index\.html$/);
  await expect(page.locator('#train')).toBeEnabled();
});
