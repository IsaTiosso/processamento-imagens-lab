"""Exporta painéis individuais (sem moldura) para os slides: apresentacao/img/*.png"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.pipeline import *  # noqa

OUT = RAIZ / "apresentacao" / "img"; OUT.mkdir(exist_ok=True)
BG, FG, MUT, AMB, BLU = "#0B1020", "#E9EDF5", "#8D97AD", "#F2B544", "#6FA8FF"


def salvar(nome, arr, cmap=None, lado=900):
    if arr.ndim == 2:
        a = arr.astype(np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        if cmap:
            arr = (plt.get_cmap(cmap)(a)[..., :3] * 255).astype(np.uint8)
        else:
            arr = (a * 255).astype(np.uint8)
            arr = np.dstack([arr] * 3)
    h, w = arr.shape[:2]; s = lado / max(h, w)
    arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / f"{nome}.jpg"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


rgb, cinza = carregar(str(RAIZ / "dados/orion.jpg"))
rgb_s, cinza_s = carregar(str(RAIZ / "dados/scorpius.jpg"))
salvar("orion_rgb", rgb, lado=1400); salvar("scorpius_rgb", rgb_s)
salvar("orion_cinza", cinza)
ec = lambda m: np.clip(m, np.percentile(m, 5), np.percentile(m, 99.97))
salvar("espectro", ec(espectro_magnitude(cinza)), "magma")
salvar("fase", espectro_fase(cinza), "twilight")
F1, F2 = np.fft.fft2(cinza.astype(float)), np.fft.fft2(cinza_s.astype(float))
salvar("troca_fase", np.clip(np.real(np.fft.ifft2(np.abs(F1) * np.exp(1j * np.angle(F2)))), 0, 255))
salvar("scorpius_cinza", cinza_s)

d0, d1 = cortes_padrao(cinza.shape)
for t, a, b in [("baixa", d1, None), ("alta", d0, None), ("banda", d0, d1)]:
    H = mascara_gaussiana((512, 512), t, a * 512 / 1280, b * 512 / 1280 if b else None)
    salvar(f"H_{t}", H, "magma", lado=500)
salvar("fundo", filtro_fft(cinza, "baixa", d0))
banda = filtro_fft(cinza, "banda", d0, d1)
salvar("banda", np.clip(banda, 0, np.percentile(banda, 99.7)))
salvar("espectro_banda", ec(espectro_magnitude(banda)), "magma")

res = pipeline(cinza)
_, b_otsu = cv2.threshold(cv2.GaussianBlur(cinza, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
salvar("bin_otsu", b_otsu); salvar("bin_fft", res["binaria"])
y0, x0, s = 380, 450, 300
salvar("zoom_rgb", rgb[y0:y0+s, x0:x0+s], lado=600)
salvar("zoom_otsu", b_otsu[y0:y0+s, x0:x0+s], lado=600)
salvar("zoom_fft", res["binaria"][y0:y0+s, x0:x0+s], lado=600)


def marcar(img, comps, n, cor=(242, 181, 68)):
    out = (img * 0.8).astype(np.uint8)
    for i, e in enumerate(comps[:n]):
        c = (int(e["x"]), int(e["y"]))
        cv2.circle(out, c, 30, cor, 4, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (c[0] + 34, c[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 1.4, cor, 3, cv2.LINE_AA)
    return out


errado = componentes(b_otsu, cinza.astype(np.float32))
salvar("top_sem", marcar(rgb, errado, 6, (255, 110, 90)))
salvar("top_com", marcar(rgb, res["componentes"], 6))
salvar("top12_orion", marcar(rgb, res["componentes"], 12), lado=1100)
salvar("top12_scorpius", marcar(rgb_s, pipeline(cinza_s)["componentes"], 12), lado=1100)
# máscara de 412 mil px destacada
m = np.zeros_like(cinza); n, rot, st, _ = cv2.connectedComponentsWithStats(b_otsu, connectivity=8)
m[rot == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])] = 255
over = (rgb * 0.55).astype(np.uint8); over[m > 0] = (0.5 * over[m > 0] + 0.5 * np.array([255, 90, 70])).astype(np.uint8)
salvar("blob_gigante", over)

crop = rgb[y0:y0+s, x0:x0+s]
px = np.float32(crop.reshape(-1, 3))
_, lab, cen = cv2.kmeans(px, 4, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 5, cv2.KMEANS_PP_CENTERS)
salvar("kmeans", np.uint8(cen)[lab.ravel()].reshape(crop.shape), lado=600)
g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
_, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
op = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
bgm = cv2.dilate(op, np.ones((3, 3), np.uint8), iterations=3)
dist = cv2.distanceTransform(op, cv2.DIST_L2, 5)
_, fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0); fg = np.uint8(fg)
_, mk = cv2.connectedComponents(fg); mk = mk + 1; mk[cv2.subtract(bgm, fg) == 255] = 0
mk = cv2.watershed(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), mk)
ws = crop.copy(); ws[cv2.dilate((mk == -1).astype(np.uint8), np.ones((2, 2))) > 0] = [255, 90, 70]
salvar("watershed", ws, lado=600)

img, v = campo_sintetico(60, gradiente=200, ruido=20, seed=7)
salvar("sint", img, lado=600)
salvar("sint_otsu", cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], lado=600)
salvar("sint_fft", pipeline(img)["binaria"], lado=600)

# ---- gráficos no tema escuro
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUT, "axes.labelcolor": FG,
                     "xtick.color": MUT, "ytick.color": MUT, "text.color": FG, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False})
cen_ = [("limpo", 0, 5), ("gradiente\nmédio", 120, 12), ("gradiente\nforte", 200, 20)]
met = [("Gauss + Otsu (original)", "#5B6478"), ("FFT + Otsu", BLU), ("FFT + k·σ (final)", AMB)]
vals = {m: [] for m, _ in met}
for _, gr, ru in cen_:
    im, vv = campo_sintetico(60, gradiente=gr, ruido=ru, seed=7)
    vals[met[0][0]].append(avaliar_deteccao(segmentar_estrelas(im), vv)["f1"])
    vals[met[1][0]].append(avaliar_deteccao(pipeline(im, metodo="otsu")["estrelas"], vv)["f1"])
    vals[met[2][0]].append(avaliar_deteccao(pipeline(im)["estrelas"], vv)["f1"])
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(3); w = 0.27
for i, (m, c) in enumerate(met):
    bars = ax.bar(x + (i - 1) * w, vals[m], w, color=c, label=m)
    for b_, val in zip(bars, vals[m]):
        ax.text(b_.get_x() + b_.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=13, color=FG)
ax.set_xticks(x, [c[0] for c in cen_]); ax.set_ylim(0, 1.12); ax.set_ylabel("F1")
ax.legend(frameon=False, fontsize=13, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
fig.tight_layout(); fig.savefig(OUT / "graf_f1.png", dpi=110); plt.close()

im, vv = campo_sintetico(60, gradiente=120, ruido=12, seed=7)
ks = [2, 3, 4, 5, 6, 8, 10, 14]
P, R = zip(*[(lambda m: (m["precisao"], m["recall"]))(avaliar_deteccao(pipeline(im, k=k)["estrelas"], vv)) for k in ks])
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(ks, P, "o-", color=BLU, lw=3, ms=8, label="precisão"); ax.plot(ks, R, "s-", color=AMB, lw=3, ms=8, label="recall")
ax.axvline(5, color=MUT, ls="--"); ax.text(5.2, 0.3, "k = 5\nescolhido", color=MUT, fontsize=13)
ax.set_xlabel("k  (limiar = mediana + k·σ)"); ax.set_ylim(0, 1.05); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "graf_k.png", dpi=110); plt.close()
print("ok", len(list(OUT.iterdir())))
