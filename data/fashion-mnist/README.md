# Fashion-MNIST — split t10k

Cópia dos arquivos originais de [Zalando Research](https://github.com/zalandoresearch/fashion-mnist), obtida em 22/09/2026. São as mesmas URLs usadas pela demonstração antes de o dataset ser incorporado ao repositório. Os arquivos gzip foram preservados sem recompressão.

- [Imagens: t10k-images-idx3-ubyte.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz)
- [Labels: t10k-labels-idx1-ubyte.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz)
- [Licença original](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE), reproduzida em [LICENSE](LICENSE).

O conjunto contém 10.000 imagens de 28 × 28 pixels, com 1.000 exemplos de cada uma das dez classes. Na demonstração, esse split é usado como fonte das amostras de treino, preservando o comportamento original.

## Integridade

SHA-256 dos arquivos comprimidos:

```text
346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073  t10k-images-idx3-ubyte.gz
67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5  t10k-labels-idx1-ubyte.gz
```

Os testes unitários conferem os hashes, a descompactação e o conteúdo IDX.

## Git LFS

A regra em `.gitattributes` rastreia os arquivos `.gz` desta pasta. Após clonar, execute `git lfs install` e `git lfs pull` se os arquivos ainda forem ponteiros de texto. Em um checkout com LFS configurado, `git add data/fashion-mnist/*.gz` armazena ponteiros no índice Git e os dados no armazenamento LFS; `git push` envia os objetos pelo hook do LFS.

A aplicação e a hospedagem estática precisam dos arquivos completos. A CI usa `actions/checkout` com `lfs: true`. Não há download alternativo do GitHub em tempo de execução.
