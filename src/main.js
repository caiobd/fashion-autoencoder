import { startApp } from './app.js';
import { getElements } from './ui/elements.js';

try {
  if (!globalThis.tf) {
    throw new Error(
      'Não foi possível carregar o TensorFlow.js. Abra esta página com internet.',
    );
  }
  await startApp(globalThis.tf, getElements());
} catch (error) {
  console.error(error);
  document.getElementById('status').textContent = error.message;
  document.getElementById('train').disabled = true;
}
