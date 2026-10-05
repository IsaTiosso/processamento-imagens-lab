# Roteiro de apresentação (meta: 12 min — janela válida 10 a 15)

Cada um fala ~4 min. Ensaiem com cronômetro: abaixo de 10 ou acima de 15 zera 1 ponto.

Slides: https://claude.ai/artifact/CbzAFG1kvVSh4iNWmUYnGP (as falas completas estão nas notas de cada slide; dá para baixar em .pptx ou PDF)

| Slides | Quem | Tempo | Conteúdo |
|---|---|---|---|
| 1–4 | Isabela | ~3:30 | capa, problema, dados, pipeline |
| 5–8 | Fernando | ~4:00 | ideia das faixas, espectro, fase, passa-faixa |
| 9–11 | Matheus | ~3:00 | segmentação, zoom, comparação de métodos |
| 12–13 | Fernando | ~1:30 | F1 no céu sintético, sensibilidade ao k |
| 14 | Isabela | ~0:45 | impacto de uma segmentação errada |
| 15–16 | Matheus | ~1:30 | limitações, conclusão |

Total ≈ 14 min: ensaiem; se passar de 14:30, encurtem o slide 11.

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
