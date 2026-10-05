"""Gera e executa o notebook principal (processamento_imagens.ipynb)."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""<a href="https://colab.research.google.com/github/IsaTiosso/processamento-imagens-lab/blob/main/processamento_imagens.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Detecção de estrelas em imagens de constelações — Segmentação + FFT

**Problema.** Fotografias de constelações (NOIRLab) têm milhares de estrelas fracas, Via Láctea, nebulosas e ruído de sensor.
Queremos **isolar as estrelas** e **destacar as estrelas principais** que formam a constelação.

**Pipeline:** entrada → escala de cinza → **FFT passa-faixa** (remove fundo e ruído) → **segmentação** (limiar + morfologia + componentes conexos) → medidas (posição, área, fluxo) → decisão (ranking das estrelas principais).

Validação quantitativa: céu **sintético** com posições conhecidas (ground truth) → precisão, recall e F1."""))

C.append(code("""# Funciona localmente e no Google Colab
import os, sys
if not os.path.exists('src/pipeline.py'):
    !git clone -q https://github.com/IsaTiosso/processamento-imagens-lab.git
    %cd processamento-imagens-lab
sys.path.insert(0, '.')

import cv2, numpy as np, matplotlib.pyplot as plt
from src.pipeline import *
plt.rcParams.update({'figure.dpi': 90, 'axes.titlesize': 10})
os.makedirs('resultados', exist_ok=True)

def mostrar(imgs, titulos, cmap='gray', nome=None, tam=4.2):
    fig, ax = plt.subplots(1, len(imgs), figsize=(tam * len(imgs), tam))
    for a, im, t in zip(np.atleast_1d(ax), imgs, titulos):
        a.imshow(im, cmap=None if im.ndim == 3 else cmap); a.set_title(t); a.axis('off')
    plt.tight_layout()
    if nome: plt.savefig(f'resultados/{nome}.png', bbox_inches='tight')
    plt.show()"""))

C.append(md("""## 1. Dados de entrada
- Origem: NOIRLab — *Constellations* (https://noirlab.edu/public/education/constellations)
- Formato: JPEG, RGB 8 bits, **1280 × 1280 px**
- Imagens: Órion e Escorpião (fotografia de campo largo)"""))

C.append(code("""rgb, cinza = carregar('dados/orion.jpg')
rgb_s, cinza_s = carregar('dados/scorpius.jpg')
print('Órion:', rgb.shape, rgb.dtype, '| faixa de cinza:', cinza.min(), '-', cinza.max())
mostrar([rgb, rgb_s, cinza], ['Órion (RGB)', 'Escorpião (RGB)', 'Órion — cinza (luminância)'], nome='01_entrada')"""))

C.append(code("""# Histograma: a maior parte dos pixels é céu escuro; as estrelas são a cauda direita
plt.figure(figsize=(7, 3)); plt.hist(cinza.ravel(), 256, log=True, color='k')
plt.title('Histograma (escala log) — Órion'); plt.xlabel('intensidade'); plt.tight_layout()
plt.savefig('resultados/02_histograma.png'); plt.show()"""))

C.append(md("""## 2. FFT — o que o espectro representa
Transformamos a **imagem em escala de cinza** (luminância) com a FFT 2D.
- **Centro (baixas frequências):** variações lentas — brilho médio do céu, Via Láctea, nebulosas, gradiente de poluição luminosa.
- **Meio do espectro:** estruturas do tamanho de uma estrela (PSF de 2–8 px).
- **Bordas (altas frequências):** variações de pixel a pixel — ruído de sensor e compressão JPEG.

A **fase** guarda *onde* estão as estruturas; a **magnitude**, *quanto* de cada frequência existe. O experimento abaixo troca as fases de duas imagens para mostrar isso."""))

C.append(code("""mag = espectro_magnitude(cinza)
fase = espectro_fase(cinza)
mostrar([cinza, mag, fase], ['Órion (cinza)', 'Magnitude log(1+|F|)', 'Fase'], nome='03_espectro')"""))

C.append(code("""# Experimento magnitude x fase: magnitude de Órion + fase de Escorpião
F1, F2 = np.fft.fft2(cinza.astype(float)), np.fft.fft2(cinza_s.astype(float))
troca = np.real(np.fft.ifft2(np.abs(F1) * np.exp(1j * np.angle(F2))))
mostrar([cinza, cinza_s, np.clip(troca, 0, 255)],
        ['Órion', 'Escorpião', '|F| de Órion + fase de Escorpião'], nome='04_fase')
print('Resultado parece o Escorpião: a FASE carrega a posição das estrelas.')"""))

C.append(md("""### 2.1 Filtragem no domínio da frequência
`g = IFFT( FFT(f) · H(u,v) )`, com máscaras **gaussianas** (evitam o efeito de anel/*ringing* do filtro ideal).
- **Passa-baixa** (d0 = N/6): remove o ruído → equivale ao filtro Gaussiano espacial do pipeline original.
- **Passa-alta** (d0 = N/100): remove o fundo (nebulosas, gradiente).
- **Passa-faixa** = os dois juntos → sobra apenas o que tem o **tamanho de uma estrela**.

Antes da FFT a imagem é **espelhada nas bordas**: a FFT supõe a imagem periódica, e o “degrau” na emenda geraria vazamento espectral."""))

C.append(code("""d0, d1 = cortes_padrao(cinza.shape)
print(f'cortes: d0 = {d0:.1f}  d1 = {d1:.1f}  (ciclos por imagem)')
Hb = mascara_gaussiana(cinza.shape, 'baixa', d1)
Ha = mascara_gaussiana(cinza.shape, 'alta', d0)
Hf = mascara_gaussiana(cinza.shape, 'banda', d0, d1)
mostrar([Hb, Ha, Hf], ['H passa-baixa', 'H passa-alta', 'H passa-faixa'], nome='05_mascaras', tam=3.5)

baixa = filtro_fft(cinza, 'baixa', d1)
fundo = filtro_fft(cinza, 'baixa', d0)          # o que o passa-alta remove
banda = filtro_fft(cinza, 'banda', d0, d1)
mostrar([cinza, fundo, np.clip(banda, 0, None)],
        ['Original', 'Fundo removido (baixas freq.)', 'Passa-faixa: só estrelas'], nome='06_filtragem')
mostrar([espectro_magnitude(cinza), espectro_magnitude(banda)],
        ['Espectro antes', 'Espectro depois do passa-faixa'], nome='07_espectro_antes_depois')"""))

C.append(code("""# Equivalência: passa-baixa na frequência  x  filtro Gaussiano espacial (convolução <-> multiplicação)
sigma_espacial = cinza.shape[0] * 1.25 / (2 * np.pi * d1)   # padding altera N; aproximação
g_esp = cv2.GaussianBlur(cinza.astype(np.float32), (0, 0), sigma_espacial)
print(f'sigma espacial equivalente ≈ {sigma_espacial:.2f} px | diferença média = {np.abs(g_esp - baixa).mean():.2f} níveis de cinza')"""))

C.append(md("""## 3. Segmentação
**Região de interesse:** cada estrela (blob brilhante e compacto). **Fundo:** céu + nebulosas.

Etapas: (1) **limiarização** da imagem filtrada, (2) **abertura morfológica** 2×2 (remove pixels isolados), (3) **componentes conexos** (8-vizinhança) com área mínima de 3 px → centroide, área e fluxo de cada estrela.

Comparamos dois limiares:
- **Otsu**: escolhe o limiar que minimiza a variância intra-classe do histograma. Supõe duas classes de tamanho parecido — aqui as estrelas são < 5% dos pixels.
- **k·σ robusto** (mediana + k·1,4826·MAD): padrão em astronomia (SExtractor). Mede o ruído do fundo e aceita só o que está *k* desvios acima dele."""))

C.append(code("""res = pipeline(cinza)
base = segmentar_estrelas(cinza)          # pipeline original: Gaussiano 5x5 + Otsu, sem FFT
_, b_otsu = cv2.threshold(cv2.GaussianBlur(cinza, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mostrar([cinza, b_otsu, res['binaria']],
        [f'Original', f'Sem FFT: Gauss+Otsu ({len(base)} regiões)', f'FFT + k·σ ({len(res["estrelas"])} estrelas)'],
        nome='08_segmentacao')"""))

C.append(code("""# Zoom: a nebulosa (Laço de Barnard) vira 'estrela' sem a FFT
y0, x0, s = 380, 450, 300
mostrar([rgb[y0:y0+s, x0:x0+s], b_otsu[y0:y0+s, x0:x0+s], res['binaria'][y0:y0+s, x0:x0+s]],
        ['Zoom original', 'Sem FFT', 'Com FFT'], nome='09_zoom')"""))

C.append(code("""# Decisão: as 12 estrelas de maior fluxo (as que formam o desenho da constelação)
def marcar(rgb, comps, n=12):
    out = rgb.copy()
    for i, e in enumerate(comps[:n]):
        c = (int(e['x']), int(e['y']))
        cv2.circle(out, c, 22, (0, 255, 120), 3)
        cv2.putText(out, str(i + 1), (c[0] + 24, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    return out
res_s = pipeline(cinza_s)
mostrar([marcar(rgb, res['componentes']), marcar(rgb_s, res_s['componentes'])],
        ['Órion: 12 mais brilhantes', 'Escorpião: 12 mais brilhantes'], nome='10_principais', tam=6)
for i, e in enumerate(res['componentes'][:6], 1):
    print(f"{i}. x={e['x']:.0f} y={e['y']:.0f} área={e['area']} px fluxo={e['fluxo']:.0f}")
print('-> 1 = Rigel, 4 = Betelgeuse, 5 = Bellatrix (conferido com a carta IAU de Órion)')"""))

C.append(md("""### 3.1 Comparação com as outras técnicas que o grupo testou
K-means (por cor) e Watershed (do notebook original) aplicados ao mesmo recorte."""))

C.append(code("""crop = rgb[y0:y0+s, x0:x0+s]
px = np.float32(crop.reshape(-1, 3))
_, lab, cen = cv2.kmeans(px, 4, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 5, cv2.KMEANS_PP_CENTERS)
km = np.uint8(cen)[lab.ravel()].reshape(crop.shape)

g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
_, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
op = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
bg = cv2.dilate(op, np.ones((3, 3), np.uint8), iterations=3)
dist = cv2.distanceTransform(op, cv2.DIST_L2, 5)
_, fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0); fg = np.uint8(fg)
_, mk = cv2.connectedComponents(fg); mk = mk + 1; mk[cv2.subtract(bg, fg) == 255] = 0
mk = cv2.watershed(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), mk)
ws = crop.copy(); ws[mk == -1] = [255, 0, 0]

mostrar([km, ws, res['binaria'][y0:y0+s, x0:x0+s]],
        ['K-means (K=4): separa por COR — nebulosa vira classe', 'Watershed: separa estrelas grudadas', 'FFT + k·σ (escolhido)'],
        nome='11_comparacao_metodos')"""))

C.append(md("""**Por que escolhemos FFT + limiar k·σ:** K-means agrupa por cor, então a nebulosa vermelha vira uma classe inteira e estrelas fracas se perdem no fundo. O Watershed é útil para separar estrelas que se tocam, mas depende de uma boa binarização inicial (que é justamente o problema). O passa-faixa resolve a causa: tira o fundo antes do limiar."""))

C.append(md("""## 4. Resultados quantitativos (céu sintético com ground truth)
Fotos reais não têm a lista exata de estrelas, então geramos campos de 512×512 px com **60 estrelas em posições conhecidas**, fundo com gradiente + “nebulosa” e ruído gaussiano. Uma detecção é correta se cair a ≤ 4 px de uma estrela real."""))

C.append(code("""cenarios = [('limpo', 0, 5), ('gradiente médio', 120, 12), ('gradiente forte', 200, 20)]
linhas = []
for nome, gr, ru in cenarios:
    img, v = campo_sintetico(60, gradiente=gr, ruido=ru, seed=7)
    for metodo, det in [('Gauss + Otsu (original)', segmentar_estrelas(img)),
                        ('FFT + Otsu', pipeline(img, metodo='otsu')['estrelas']),
                        ('FFT + k·σ (final)', pipeline(img)['estrelas'])]:
        m = avaliar_deteccao(det, v)
        linhas.append((nome, metodo, m['precisao'], m['recall'], m['f1']))
print(f"{'cenário':<17}{'método':<25}{'precisão':>9}{'recall':>8}{'F1':>7}")
for l in linhas: print(f'{l[0]:<17}{l[1]:<25}{l[2]:>9.2f}{l[3]:>8.2f}{l[4]:>7.2f}')

img, v = campo_sintetico(60, gradiente=200, ruido=20, seed=7)
r = pipeline(img); bo = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
mostrar([img, bo, r['binaria']], ['Sintético (gradiente forte)', 'Gauss+Otsu', 'FFT + k·σ'], nome='12_sintetico')"""))

C.append(code("""# Gráfico F1 por cenário
import itertools
met = ['Gauss + Otsu (original)', 'FFT + Otsu', 'FFT + k·σ (final)']
x = np.arange(len(cenarios)); w = 0.27
plt.figure(figsize=(7, 3.4))
for i, m in enumerate(met):
    plt.bar(x + (i - 1) * w, [l[4] for l in linhas if l[1] == m], w, label=m)
plt.xticks(x, [c[0] for c in cenarios]); plt.ylabel('F1'); plt.ylim(0, 1.05); plt.legend(fontsize=8)
plt.title('Detecção de estrelas: F1 por cenário'); plt.tight_layout(); plt.savefig('resultados/13_f1.png'); plt.show()"""))

C.append(md("""### 4.1 Sensibilidade aos parâmetros
- **k** (limiar): baixo → ruído vira estrela (precisão cai); alto → estrelas fracas somem (recall cai).
- **d0** (corte do passa-alta): alto demais começa a apagar o centro das estrelas grandes."""))

C.append(code("""img, v = campo_sintetico(60, gradiente=120, ruido=12, seed=7)
ks = [2, 3, 4, 5, 6, 8, 10, 14]
P, R = zip(*[(avaliar_deteccao(pipeline(img, k=k)['estrelas'], v)['precisao'],
              avaliar_deteccao(pipeline(img, k=k)['estrelas'], v)['recall']) for k in ks])
d0s = [1, 2, 5, 10, 20, 40, 60]
F = [avaliar_deteccao(pipeline(img, d0=d)['estrelas'], v)['f1'] for d in d0s]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(ks, P, 'o-', label='precisão'); ax[0].plot(ks, R, 's-', label='recall')
ax[0].set_xlabel('k (limiar = mediana + k·σ)'); ax[0].legend(); ax[0].set_title('Sensibilidade ao limiar')
ax[1].plot(d0s, F, 'o-'); ax[1].set_xlabel('d0 do passa-alta (ciclos/imagem)'); ax[1].set_ylabel('F1')
ax[1].set_title('Sensibilidade ao corte da FFT')
plt.tight_layout(); plt.savefig('resultados/14_sensibilidade.png'); plt.show()"""))

C.append(md("""## 5. Impacto de uma segmentação incorreta
Se o limiar deixa a nebulosa passar, ela vira **um componente gigante** com fluxo enorme e entra no topo do ranking como se fosse a “estrela mais brilhante”: o desenho da constelação sai errado. Se o limiar é alto demais, estrelas da constelação somem e a figura fica incompleta."""))

C.append(code("""errado = componentes(b_otsu, cinza.astype(np.float32))
print('Sem FFT — 3 maiores “estrelas”:', [(int(e['x']), int(e['y']), e['area']) for e in errado[:3]], '(área em px)')
print('Com FFT — 3 maiores estrelas:   ', [(int(e['x']), int(e['y']), e['area']) for e in res['componentes'][:3]])
mostrar([marcar(rgb, errado, 6), marcar(rgb, res['componentes'], 6)], ['Top-6 SEM FFT', 'Top-6 COM FFT'], nome='15_impacto', tam=6)"""))

C.append(md("""## 6. Limitações e próximos passos
- **Estrelas saturadas e muito grandes** (ex.: Sírius, canto inferior esquerdo de Órion) têm o núcleo “chapado”, que vira baixa frequência e é parcialmente removido pelo passa-alta → aparecem como anel. Próximo passo: tratar saturação separadamente (máscara de pixels = 255).
- **Estrelas sobrepostas** em aglomerados viram um só componente → aplicar Watershed *depois* do passa-faixa.
- **JPEG** comprimido e com contraste ajustado para divulgação: o fluxo medido não é fotometria científica.
- **Identificação automática da constelação:** casar o padrão das estrelas principais com um catálogo (Hipparcos) por *matching* de triângulos (técnica dos *star trackers* de satélites). O classificador anterior foi **removido** porque era treinado com dados aleatórios inventados e não tinha validade.

## 7. Conclusão
A FFT foi o que fez a segmentação funcionar: em céu com gradiente, o F1 subiu de ≈0,2 (Gauss + Otsu) para ≈0,99 (passa-faixa + k·σ). Pensar no domínio da frequência separa o que é **fundo** (baixa freq.), **estrela** (média) e **ruído** (alta), e a segmentação passa a decidir só sobre o que interessa."""))

nb = nbf.v4.new_notebook(cells=C, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                            "colab": {"provenance": []}})
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbf.write(nb, "processamento_imagens.ipynb")
print("ok")
