# Autoencoder Fashion-MNIST

Demonstração para ensinar embeddings: um autoencoder aprende a reconstruir peças de roupa, enquanto a interface mostra uma projeção PCA do gargalo. É possível explorar pontos, comparar reconstruções e interpolar entre duas amostras.

## Executar

Requer Node.js 22 ou superior, Git LFS para obter o dataset ao clonar o repositório e um navegador com suporte a módulos JavaScript e `DecompressionStream`.

```sh
git lfs install
git lfs pull
npm start
```

Abra **http://127.0.0.1:8000**. Não é necessário instalar dependências npm para executar a aplicação. O Fashion-MNIST é servido a partir dos arquivos locais do repositório; a primeira abertura ainda precisa de internet para carregar TensorFlow.js 4.22.0 pelo CDN. Use um servidor HTTP; abrir `index.html` diretamente por `file://` não permite carregar os módulos em todos os navegadores.

A página também pode ser publicada em qualquer hospedagem estática, copiando `index.html`, `src/`, `styles/` e `data/`. Execute `git lfs pull` antes de copiar os arquivos: a publicação precisa conter os arquivos `.gz` completos, não os ponteiros de texto do LFS. O arquivo HTML com o nome antigo redireciona para `index.html` e pode ser incluído para preservar links existentes.

## Organização

```text
index.html                   Estrutura e textos da interface
styles/main.css              Aparência e layout responsivo
data/fashion-mnist/           Dataset em Git LFS, origem, hashes e licença
src/
  main.js                    Inicialização do navegador
  app.js                     Carregamento, criação, treino e controles
  data/fashion-mnist.js      Download, leitura IDX e seleção de amostras
  model/
    autoencoder.js           Rede, treino, inferência e descarte dos tensores
    gelu.js                  Ativação GELU
    learning-rate.js         Decaimento cosseno da taxa de aprendizado
  math/projection.js         PCA, alinhamento e inversa da projeção
  ui/
    elements.js              Elementos esperados no HTML
    canvas.js                Desenho de imagens e resolução do canvas
    canvas-navigation.js     Gestos de mouse e toque
    embedding-plot.js        Desenho das miniaturas, zoom e deslocamento
    embedding-explorer.js    Seleção, exploração e interpolação
scripts/serve.js              Servidor local de desenvolvimento
tests/unit/                  Dados, matemática e contrato do modelo
tests/browser/               Fluxos completos com Chromium e dados reais
```

Os módulos usam JavaScript nativo, sem etapa de build. As regras de treino ficam no modelo; o aplicativo controla seu ciclo de vida; os módulos de interface cuidam dos elementos e canvases. PCA e leitura de IDX podem ser usados sem DOM ou TensorFlow.

## Modelo e dados

A arquitetura foi preservada: **784 → 128 (GELU) → gargalo linear → 128 (GELU) → 784 (sigmoid)**. O gargalo varia de 2 a 32 dimensões. O treino usa Adam, binary cross-entropy, batches de até 32 amostras, inicialização Glorot normal com seed 42 e taxa de aprendizado com decaimento cosseno de 0,002 a 0,00005. A ordem das amostras não é embaralhada.

O repositório inclui os mesmos arquivos do [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): `t10k-images-idx3-ubyte.gz` e `t10k-labels-idx1-ubyte.gz`, em `data/fashion-mnist/`, rastreados pelo Git LFS. O navegador os carrega da própria aplicação, sem consultar o GitHub. A [documentação do dataset](data/fashion-mnist/README.md) registra a origem, os hashes SHA-256 e a licença. Esse split tem 10 mil imagens de 28 × 28 pixels. Aqui ele é usado para a demonstração de treino, como na versão original; a aplicação não faz uma avaliação separada de generalização.

Os pixels são normalizados para `[0, 1]`. A seleção alterna entre as dez classes e usa passo determinístico 37 dentro de cada classe. Os labels servem para selecionar e colorir as amostras, sem participar da função de perda. A visualização acompanha até 80 amostras, aplica PCA e alinha as projeções sucessivas por rotação para reduzir mudanças de orientação durante o treino.

## Contrato do autoencoder

`src/model/autoencoder.js` concentra a implementação que poderá virar exercício na próxima etapa. **Nesta versão, ela está completa.**

```js
const model = new Autoencoder(tf, 16);
const embeddings = await model.encode(images); // N vetores de 16 números
const reconstruction = await model.reconstruct(images[0]); // 784 pixels
const generated = await model.decode(embeddings[0]); // 784 pixels

await model.train(images, {
  epochs: 100,
  onEpochBegin: (epoch, learningRate) => {},
  onEpochEnd: (epoch, loss) => {},
});

model.dispose();
```

`images` é um array de imagens, cada uma um `Float32Array` de 784 pixels. `encode` devolve arrays de números; `reconstruct` e `decode` devolvem `Float32Array`. Os callbacks de treino podem ser assíncronos. `stop()` solicita interrupção ao final da época atual; outra chamada a `train()` continua com os pesos existentes.

O modelo é dono dos tensores e do otimizador. A interface aguarda as leituras pendentes antes de chamar `dispose()` ao recriar a rede. O encoder e o modelo completo compartilham pesos, que são descartados uma única vez. A interpolação acontece nos vetores completos do gargalo; clicar no espaço vazio reconstrói um vetor restrito ao plano do PCA.

## Desenvolvimento e verificações

```sh
npm ci
npm run check             # ESLint, formatação e testes unitários
npx playwright install chromium
npm run test:browser      # Fluxos no navegador com os arquivos reais do Fashion-MNIST
```

Se já tiver Chromium instalado, use `CHROMIUM_PATH=/caminho/para/chromium npm run test:browser`. Com as dependências, o Chromium e os objetos LFS presentes, os testes não precisam de rede externa: o dataset vem do servidor local e o TensorFlow.js vem da dependência local, na mesma versão do CDN. A CI faz checkout com `lfs: true`.

Os testes verificam seleção e validação do dataset, projeção e inversa do PCA, equivalência entre reconstruir e executar encoder/decoder, interrupção e retomada do treino e liberação dos tensores. Os fluxos no navegador cobrem exploração, interpolação, zoom, treino, recriação, tela pequena e o endereço antigo.

Use `npm run format` para aplicar a formatação. O workflow em `.github/workflows/check.yml` executa essas verificações em pushes e pull requests.
