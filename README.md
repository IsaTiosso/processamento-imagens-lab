

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IsaTiosso/processamento-imagens-lab/blob/main/processamento_imagens.ipynb)


# Detecção de Estrelas em Imagens de Constelações (NOIRLab)

Projeto desenvolvido para a disciplina Processamento de Imagens e Sinais, sob a orientação do Prof. Dr. Vinicius Santos Andrade.

# 1. Integrantes da Equipe

Fernando Fleuri Barbosa

Isabela Xavier Tiosso

Matheus Eduardo Nunhez

# 2. Descrição Detalhada do Projeto e Pipeline de Processamento de Imagens

O objetivo principal deste projeto é isolar estrelas em imagens de campo largo de constelações e destacar as estrelas principais que desenham o padrão estelar. Fotografias reais do céu profundo contêm múltiplos desafios: milhares de estrelas fracas, a presença da Via Láctea, nebulosas difusas, gradientes de luminosidade de fundo e ruído oriundo do sensor e de compressão JPEG (onde menos de 5% dos pixels representam estrelas de fato).

Para solucionar esses desafios, o projeto implementa um pipeline de processamento de imagens em 5 etapas principais:

- Pré-processamento (Conversão de Entrada):

A imagem colorida de entrada (formato RGB, 8 bits por canal, $1280 \times 1280$ pixels) é convertida para escala de cinza utilizando canais de luminância para padronização analítica.

- Transformação de Frequência (FFT 2D):

A imagem em escala de cinza é mapeada para o domínio de frequência utilizando a Transformada Rápida de Fourier 2D.

- Permite separar os elementos visuais por suas escalas espaciais características:

Baixa frequência: Fundo, Via Láctea e nebulosas difusas.

Média frequência (Anel do meio): Estrelas pontuais (tamanhos típicos entre 2 a 8 pixels).

Alta frequência: Ruído pixel a pixel e artefatos de compressão JPEG.

- Filtragem Passa-Faixa Gaussiano:

Aplicação de um filtro gaussiano passa-faixa no domínio de frequência para reter estritamente o anel de interesse (estrelas), descartando o fundo e o ruído de alta frequência.

Utiliza-se espelhamento de bordas para prevenir vazamentos espectrais (ringing), preservando a magnitude filtrada e mantendo a fase intacta (essencial, pois a fase retém a localização espacial exata das estruturas). A operação no domínio da frequência equivale a uma convolução espacial otimizada.

- Transformada Inversa (IFFT) e Segmentação:

Retorno ao domínio espacial (IFFT) livre do fundo e do ruído.

Aplicação de limiarização estatística robusta baseada na mediana e desvio absoluto da mediana, que supera métodos globais como Otsu (ineficazes em imagens onde estrelas ocupam <5% da área).

Morfologia matemática: Abertura morfológica para eliminação de pixels isolados.

Rotulagem de componentes conexos: Identificação de componentes 8-conexos com área mínima de 3 pixels.

- Decisão, Extração de Características e Ranking:

Cálculo de centróides, área e fluxo de cada estrela detectada.

Ordenação por fluxo luminoso para gerar o ranking das estrelas principais (top-12), permitindo identificar corretamente objetos de referência como Rigel, Betelgeuse e Bellatrix (em Órion), mitigando falhas clássicas onde o fundo da Via Láctea era erroneamente interpretado como uma única estrela gigante de centenas de milhares de pixels.

## Tabela da Pipeline 

```
imagem RGB ─► escala de cinza ─► FFT 2D ─► filtro passa-faixa gaussiano ─► IFFT
                                          (tira fundo = baixa freq.
                                           e ruído   = alta freq.)
   ─► limiar k·σ robusto ─► abertura morfológica ─► componentes conexos
   ─► centroide, área, fluxo de cada estrela ─► ranking das estrelas principais
```

| Etapa | Técnica | Parâmetros |
|---|---|---|
| Pré-processamento | RGB → cinza (luminância) | — |
| **FFT** | passa-faixa gaussiano, com espelhamento de bordas | d0 = N/100, d1 = N/6 (ciclos/imagem) |
| **Segmentação** | limiar mediana + k·1,4826·MAD; abertura 2×2; componentes 8-conexos | k = 5, área mín. = 3 px |
| Decisão | ordenação por fluxo | top-12 |

# 3. Integração com Inteligência Artificial

É importante destacar que não houve uma integração oficial ou curricular com a disciplina de Inteligência Artificial neste projeto, sendo o escopo desenvolvido inteiramente sob a ótica clássica de Processamento de Sinais e Visão Computacional.

No entanto, ferramentas e conceitos de IA foram empregados de forma pontual e auxiliar ao longo do desenvolvimento, embora sem exercer um papel central na arquitetura do sistema:

Auxílio na Programação e Depuração: Modelos de linguagem e assistentes baseados em IA foram utilizados esporadicamente como suporte para estruturar trechos de código em Python, debugar funções numéricas (como a aplicação da FFT e cálculos de MAD) e otimizar a visualização de gráficos com a biblioteca matplotlib.

Exploração de Abordagens Alternativas: Nos estágios iniciais de pesquisa (registrados na pasta legado/), foram testados conceitos fundamentais de aprendizado não supervisionado (como agrupamento via K-means) para segmentação de cores. Contudo, constatou-se que abordagens puramente estatísticas e baseadas em frequência (como o filtro passa-faixa e o limiar robusto) apresentavam resultados analíticos superiores e mais determinísticos para o problema astronômico em questão, tornando o uso de modelos complexos de IA dispensável para o pipeline final.

# 4. Requisitos do Sistema (requirements.txt)

O arquivo de dependências do projeto encontra-se estruturado da seguinte forma:

```
numpy>=1.20.0
scipy>=1.6.0
matplotlib>=3.3.0
scikit-image>=0.18.0
pytest>=6.0.0

```

# 5. Dataset Utilizado

As imagens de campo largo utilizadas para validação e testes práticos do pipeline foram obtidas através do repositório educacional oficial do NOIRLab:

Link do Dataset: [NOIRLab — Constellations](https://noirlab.edu/public/education/constellations)

Amostras focadas nas constelações de Órion e Escorpião.

# Estrutura do Repositório

```
processamento_imagens.ipynb — Notebook principal de execução (compatível com Google Colab ou execução local).

src/pipeline.py — Módulo contendo as funções encapsuladas do pipeline de processamento.

tests/ — Conjunto de testes unitários automatizados (python -m pytest).

dados/ — Imagens do NOIRLab utilizadas (Órion, Escorpião).

legado/ — Notebooks exploratórios iniciais (comparações com Otsu puro, K-means e Watershed).

construir_notebook.py — Script automatizado para regeneração e execução do notebook.

apresentacao/ — Slides e roteiro da apresentação do projeto.

```

# Como Executar

Para rodar o projeto localmente, execute os seguintes comandos no terminal:

```
pip install -r requirements.txt
python -m pytest
python construir_notebook.py

```



