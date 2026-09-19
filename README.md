**Processamento de Imagens de Constelações (NOIRLab)**


Repositório desenvolvido para a atividade prática de processamento e análise de imagens astronômicas utilizando Python e Google Colab, integrando conceitos fundamentais de Visão Computacional e Processamento de Imagens.

1. Integrantes da Equipe
* Fernando Fleuri Barbosa
* Isabela Xavier Tiosso
* Matheus Eduardo Nunhez

2. Descrição Detalhada do Projeto e Pipeline de Processamento

O objetivo deste projeto é aplicar um pipeline de tratamento digital de imagens em fotografias de constelações celestes para mitigar ruídos de sensores e destacar elementos estruturais (estrelas e linhas de contorno).



---


3. Pipeline de Processamento de Imagem:

->  **Aquisição e Entrada (`Google Colab Files`):** Carregamento dinâmico de imagens via upload do usuário no ambiente em nuvem do Colab.


-> **Conversão de Espaço de Cores (`BGR para RGB`):** O OpenCV carrega as imagens nativamente no padrão BGR. Logo na entrada, realizamos a conversão para RGB para exibição correta e manipulação dos canais visuais.

->  **Suavização e Redução de Ruído (`Filtro Gaussiano`):** Aplicação de uma matriz de convolução gaussiana ($5 \times 5$) para atenuar ruídos de alta frequência gerados por sensores fotográficos e preparar o sinal para as etapas de segmentação.

-> **Segmentação na Escala de Cinza e Limiarização de Otsu (`Thresholding`):** Conversão da imagem suavizada para escala de cinza seguida pelo método estatístico de Otsu, que analisa o histograma para calcular automaticamente o limiar ideal de separação, isolando os pontos de alta intensidade luminosa (estrelas e traços) do fundo escuro.

-> **Segmentação de Cores com K-means:** Aplicação do algoritmo de aprendizado não supervisionado *K-means Clustering* para agrupar os pixels com base em suas características cromáticas e de intensidade, segmentando a imagem em regiões distintas (ex: fundo celeste versus elementos em destaque).

-> **Segmentação com o Algoritmo Watershed:** Utilização da técnica baseada em morfologia matemática e linhas divisórias de águas (*Watershed*) para separar objetos sobrepostos ou contornar de forma precisa as estruturas das constelações e aglomerados estelares.

---

4. Integração com IA e Visão Computacional

Sob a ótica de engenharia de software e inteligência artificial, este pipeline serve como a camada essencial de **pré-processamento (*data preprocessing*)** e **extração de características (*feature extraction*)**. 

Em arquiteturas mais complexas de Visão Computacional (como Redes Neurais Convolucionais - CNNs), imagens limpas por meio de filtragem espacial e binarização otimizada reduzem drasticamente o ruído de fundo, permitindo que o modelo aprenda padrões geométricos e constelações com maior acurácia e menor custo computacional.

O pipeline de processamento de imagens desenvolvido atua diretamente como a etapa fundamental de **Engenharia de Atributos (*Feature Engineering*)** e **Pré-processamento de Dados** para a Inteligência Artificial:

1. **Redução de Dimensionalidade e Ruído:** As técnicas de Filtro Gaussiano, Otsu, K-means e Watershed removem a poluição visual do fundo celeste e isolam os pontos de interesse (estrelas e linhas).

2. **Alimentação do Modelo:** Em uma etapa posterior de IA (como a utilização de Redes Neurais Convolucionais ou classificadores baseados em Machine Learning), essas imagens tratadas otimizam a convergência do modelo, permitindo que o algoritmo aprenda os padrões geométricos das constelações com maior acurácia e menor custo computacional.

---

5. Arquivo de Requerimentos (`requirements.txt`)

As dependências utilizadas no projeto constam abaixo:

```text
numpy==1.26.4
matplotlib==3.8.3
opencv-python==4.9.0.80
scikit-learn==1.4.1.post1
