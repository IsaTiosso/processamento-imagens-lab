# Roteiro de apresentação (meta: 12 min — janela válida 10 a 15)

Cada um fala ~4 min. Ensaiem com cronômetro: abaixo de 10 ou acima de 15 zera 1 ponto.

| # | Slide | Quem | Tempo | Pontos-chave da fala |
|---|---|---|---|---|
| 1 | Título | Isabela | 0:20 | Nomes, tema. |
| 2 | Problema | Isabela | 1:10 | Foto do céu = milhares de estrelas fracas + Via Láctea + nebulosas + ruído. Queremos as estrelas, e destacar as principais. Uso real: *star trackers* de satélites fazem exatamente isso para se orientar. |
| 3 | Dados | Isabela | 1:00 | NOIRLab, JPEG RGB 1280×1280, 8 bits. Histograma: quase tudo é céu escuro, estrelas são poucos pixels na cauda direita → isso vai importar na escolha do limiar. |
| 4 | Pipeline | Isabela | 1:30 | Ler as caixas da esquerda para a direita. Frase-chave: “cada etapa depende da anterior; se a FFT não tira o fundo, o limiar marca a Via Láctea como estrela”. |
| 5 | FFT espectro | Fernando | 1:40 | Transformamos a imagem em cinza. Centro do espectro = baixa frequência = fundo/nebulosa; meio = estrelas; bordas = ruído. Magnitude = quanto; fase = onde. Experimento: magnitude de Órion + fase de Escorpião → a imagem parece o Escorpião (faixas escuras, Antares). |
| 6 | Passa-faixa | Fernando | 1:40 | g = IFFT(FFT·H). Gaussiano para não ter *ringing*. d0 tira fundo, d1 tira ruído. Mostrar o “fundo removido” e o resultado. Espelhamento das bordas: a FFT acha que a imagem se repete; sem isso aparece borda falsa. |
| 7 | Segmentação | Matheus | 1:30 | ROI = cada estrela. Limiar k·σ robusto, abertura, componentes conexos. Sem FFT: 6517 regiões, metade da imagem branca. Otsu falha porque supõe duas classes de tamanho parecido. |
| 8 | Comparação | Matheus | 1:00 | K-means separa por cor → nebulosa vira classe. Watershed separa estrelas grudadas mas precisa de boa binarização. FFT ataca a causa. |
| 9 | Resultados | Fernando | 1:00 | Céu sintético com 60 estrelas conhecidas: F1 0,17 → 0,99. Gráfico de k: precisão × recall. |
| 10 | Impacto | Isabela | 0:40 | Sem FFT, a “estrela nº 1” é a Via Láctea (412 mil px). Com FFT: Rigel, Betelgeuse, Bellatrix. |
| 11 | Limitações | Matheus | 0:50 | Sírius saturada vira anel; aglomerados; JPEG. Próximo passo: casar triângulos com catálogo. Classificador antigo removido (dados aleatórios). |
| 12 | Conclusão | Matheus | 0:40 | Fundo = baixa, estrela = média, ruído = alta. |

Total ≈ 13:40 com transições → cortem ou estiquem nos slides 5–9.

## Perguntas prováveis do professor (todos devem saber)

**Por que passa-faixa e não só passa-alta?** O passa-alta tira o fundo mas deixa o ruído de pixel, que vira milhares de “estrelas” falsas. O passa-baixa limita isso.

**Como escolheram d0 e d1?** Pelo tamanho das estruturas. Uma estrutura de tamanho L px corresponde a ~N/L ciclos por imagem. Fundo/nebulosa > N/4 px → abaixo de ~N/100; ruído = 1 px → acima de ~N/6. Validado no gráfico de sensibilidade do notebook.

**Por que gaussiano e não ideal?** O filtro ideal tem corte abrupto; na imagem isso vira oscilação (*ringing*) em volta das estrelas (a transformada de um degrau é uma sinc).

**O que é o espelhamento das bordas?** A DFT supõe a imagem periódica. Se o fundo é mais claro de um lado, a emenda cria um degrau → energia espalhada em todas as frequências (vazamento espectral). Espelhar torna a emenda contínua.

**Por que o centro do espectro é tão brilhante?** É o componente DC = brilho médio da imagem; usamos log(1+|F|) para conseguir ver o resto.

**Por que k·σ e não Otsu?** Otsu minimiza a variância intra-classe e funciona bem quando o histograma é bimodal com classes parecidas. Estrelas são < 5% dos pixels; depois da FFT o fundo vira ruído em torno de zero e o Otsu cai dentro do ruído (F1 0,03 no cenário forte). k·σ mede o ruído do fundo (MAD, robusto às próprias estrelas) e aceita só o que está 5 σ acima.

**Por que 1,4826?** Converte MAD em desvio-padrão para ruído gaussiano.

**Como sabem que a segmentação está certa?** Fotos reais não têm gabarito, então geramos céu sintético com posições conhecidas e medimos precisão/recall/F1 (acerto ≤ 4 px). Na foto real conferimos com a carta IAU: Rigel, Betelgeuse e Bellatrix no topo.

**Qual a relação entre o filtro Gaussiano espacial e a FFT?** Teorema da convolução: convoluir no espaço = multiplicar na frequência. A transformada de uma gaussiana é uma gaussiana; σ_freq ≈ N / (2π·σ_espaço).

**E a fase?** No experimento de troca, a imagem resultante tem a estrutura da imagem cuja fase usamos → a fase codifica a posição das estrelas e bordas.

**O que acontece se a segmentação errar?** Slide 10: o componente gigante entra no topo do ranking e o desenho da constelação sai errado; limiar alto demais → estrelas somem.

**Onde está o código?** `src/pipeline.py`; notebook `processamento_imagens.ipynb` (abre no Colab pelo botão do README).
