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

## Gerar um HTML único

```sh
npm ci
git lfs pull
npm run build
```

O comando gera **`dist/autoencoder.html`**, com cerca de 7,5 MiB. Esse arquivo inclui a interface, os estilos, o JavaScript, o TensorFlow.js e os dois arquivos gzip do Fashion-MNIST. Pode ser aberto diretamente no navegador, inclusive por `file://`, sem servidor, instalação ou conexão com a internet. O navegador ainda precisa oferecer `DecompressionStream`.

Depois de instalar as dependências e obter os objetos LFS, o próprio build também funciona sem internet. Os arquivos do dataset são incorporados sem recompressão, como data URLs. O esbuild reúne os módulos JavaScript e inclui a versão do TensorFlow.js instalada pelo `package-lock.json`. As licenças do TensorFlow.js e do dataset ficam embutidas no HTML, sem alterar a interface.

Continue editando os arquivos de `src/`, `styles/` e `index.html`, e rode `npm run build` novamente para atualizar a distribuição. `dist/` é gerado e ignorado pelo Git. Compartilhe apenas `dist/autoencoder.html`; os destinatários não precisam do restante do repositório.

## Organização

```text
index.html                   Estrutura e textos da interface
styles/main.css              Aparência e layout responsivo
data/fashion-mnist/           Dataset em Git LFS, origem, hashes e licença
src/
  main.js                    Inicialização do navegador
  app.js                     Carregamento, criação, treino e controles
  data/fashion-mnist.js       Carregamento, leitura IDX e seleção de amostras
  data/dataset-urls.js        Localização do dataset; embutido no build
  model/
    architecture.js          Definição da rede: arquivo de trabalho dos alunos
    autoencoder.js           Treino, inferência e descarte dos tensores
    learning-rate.js         Decaimento cosseno da taxa de aprendizado
  math/projection.js         PCA, alinhamento e inversa da projeção
  ui/
    elements.js              Elementos esperados no HTML
    canvas.js                Desenho de imagens e resolução do canvas
    canvas-navigation.js     Gestos de mouse e toque
    embedding-plot.js        Desenho das miniaturas, zoom e deslocamento
    embedding-explorer.js    Seleção, exploração e interpolação
scripts/serve.js              Servidor local de desenvolvimento
scripts/build.js              Geração do HTML único com esbuild
licenses/                     Licença do TensorFlow.js incluída na distribuição
dist/autoencoder.html         HTML independente gerado (ignorado pelo Git)
tests/unit/                  Dados, matemática e contrato do modelo
tests/browser/               Fluxos completos com Chromium e dados reais
```

O desenvolvimento usa módulos JavaScript nativos, sem exigir build. A geração do HTML único é opcional. As regras de treino ficam no modelo; o aplicativo controla seu ciclo de vida; os módulos de interface cuidam dos elementos e canvases. PCA e leitura de IDX podem ser usados sem DOM ou TensorFlow.

## Modelo e dados

A arquitetura é **784 → 128 (ReLU) → gargalo linear → 128 (ReLU) → 784 (sigmoid)**. O gargalo varia de 2 a 32 dimensões. As ativações são nativas do TensorFlow.js e configuradas diretamente nas camadas densas. O treino usa Adam, binary cross-entropy, batches de até 32 amostras, inicialização padrão do TensorFlow.js sem seed fixa e taxa de aprendizado com decaimento cosseno de 0,002 a 0,00005. A ordem das amostras não é embaralhada.

O repositório inclui os mesmos arquivos do [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): `t10k-images-idx3-ubyte.gz` e `t10k-labels-idx1-ubyte.gz`, em `data/fashion-mnist/`, rastreados pelo Git LFS. O navegador os carrega da própria aplicação, sem consultar o GitHub. A [documentação do dataset](data/fashion-mnist/README.md) registra a origem, os hashes SHA-256 e a licença. Esse split tem 10 mil imagens de 28 × 28 pixels. Aqui ele é usado para a demonstração de treino, como na versão original; a aplicação não faz uma avaliação separada de generalização.

Os pixels são normalizados para `[0, 1]`. A seleção alterna entre as dez classes e usa passo determinístico 37 dentro de cada classe. Os labels servem para selecionar e colorir as amostras, sem participar da função de perda. A visualização acompanha até 80 amostras, aplica PCA e alinha as projeções sucessivas por rotação para reduzir mudanças de orientação durante o treino.

## Onde os alunos editam o modelo

A definição da rede está em [`src/model/architecture.js`](src/model/architecture.js). Esse é o arquivo de trabalho dos alunos. **A implementação continua completa nesta versão.**

Usamos a [API Layers do TensorFlow.js, inspirada no Keras](https://www.tensorflow.org/js/guide/layers_for_keras_users), para manter a execução no navegador. O encoder e o decoder são modelos `tf.sequential()` com listas explícitas de camadas `tf.layers`. Cada lista começa com `tf.layers.inputLayer()`: a entrada do encoder tem 784 pixels, e a do decoder tem `embeddingSize` valores. A API funcional `tf.model()` compõe os dois em um autoencoder com pesos compartilhados.

Para experimentar outra arquitetura, altere as camadas nas listas `layers`: quantidade de unidades, ativações ou camadas intermediárias. Para trocar uma ativação, edite a propriedade `activation` da camada densa. Preserve o contrato de entrada e saída:

- O encoder recebe 784 pixels e devolve `embeddingSize` valores.
- O decoder recebe `embeddingSize` valores e devolve 784 pixels em `[0, 1]`.
- O modelo completo usa as mesmas instâncias de encoder e decoder, para que o treino atualize os pesos usados na exploração e na interpolação.

O arquivo de definição não trata de DOM, dataset, otimizador, callbacks de treino ou gerenciamento de tensores. Esses detalhes ficam em `autoencoder.js` e nos demais módulos. O diagrama do HTML ilustra a arquitetura de referência 784 → 128 → gargalo → 128 → 784; ele não é gerado automaticamente a partir das camadas.

É possível inspecionar a definição diretamente, sem inicializar a aplicação:

```js
import { createAutoencoderModels } from './src/model/architecture.js';

const { encoder, decoder, model } = createAutoencoderModels(tf, 16);
encoder.summary();
decoder.summary();
model.summary();
model.dispose(); // Também libera os submodelos compartilhados.
```

## Contrato do autoencoder

A classe `Autoencoder`, em `src/model/autoencoder.js`, adapta os modelos para a interface e mantém as operações de treino e inferência:

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

O modelo é dono dos tensores e do otimizador. A interface aguarda as leituras pendentes antes de chamar `dispose()` ao recriar a rede. O modelo composto descarta os submodelos encoder e decoder; não os descarte novamente. A interpolação acontece nos vetores completos do gargalo; clicar no espaço vazio reconstrói um vetor restrito ao plano do PCA.

## Desenvolvimento e verificações

```sh
npm ci
npm run check             # ESLint, formatação e testes unitários
npx playwright install chromium
npm run test:browser      # Gera o HTML e testa os modos servidor e arquivo offline
```

Se já tiver Chromium instalado, use `CHROMIUM_PATH=/caminho/para/chromium npm run test:browser`. Com as dependências, o Chromium e os objetos LFS presentes, os testes não precisam de rede externa: o dataset vem do servidor local e o TensorFlow.js vem da dependência local, na mesma versão do CDN. A CI faz checkout com `lfs: true`.

Os testes verificam seleção e validação do dataset, projeção e inversa do PCA, equivalência entre reconstruir e executar encoder/decoder, interrupção e retomada do treino e liberação dos tensores. Os fluxos no navegador cobrem exploração, interpolação, zoom, treino, recriação, tela pequena e o endereço antigo. O mesmo fluxo principal é executado no HTML único, copiado para outra pasta e aberto por `file://` com a rede desativada, verificando que nenhum arquivo auxiliar é solicitado.

Use `npm run format` para aplicar a formatação. O workflow em `.github/workflows/check.yml` executa essas verificações em pushes e pull requests.
