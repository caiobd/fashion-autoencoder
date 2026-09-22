/** Centraliza o contrato entre o HTML e os módulos da interface. */
export function getElements() {
  const ids = [
    'status',
    'latent',
    'latentLabel',
    'latentValue',
    'samples',
    'epochs',
    'reset',
    'train',
    'stop',
    'next',
    'epoch',
    'loss',
    'lr',
    'inputCanvas',
    'outputCanvas',
    'embeddingCanvas',
    'latentInputCanvas',
    'latentCanvas',
    'latentInputCaption',
    'latentOutputCaption',
    'latentInfo',
    'exploreModeBtn',
    'interpolateModeBtn',
    'clearSelectionBtn',
    'explorePanel',
    'interpolatePanel',
    'interpACanvas',
    'interpBCanvas',
    'interpResultCanvas',
    'interpSlider',
    'interpWeight',
    'interpPlayBtn',
    'interpInfo',
    'zoomIn',
    'zoomOut',
    'zoomReset',
    'zoomReadout',
  ];
  const elements = Object.fromEntries(
    ids.map((id) => {
      const element = document.getElementById(id);
      if (!element) throw new Error(`Elemento obrigatório ausente: #${id}`);
      return [id, element];
    }),
  );
  elements.zoomInBtn = elements.zoomIn;
  elements.zoomOutBtn = elements.zoomOut;
  elements.zoomResetBtn = elements.zoomReset;
  return elements;
}
