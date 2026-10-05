"""Pipeline de detecção de estrelas em imagens de constelações.

Etapas:
    1. Entrada: imagem RGB -> escala de cinza (luminância)
    2. FFT: filtro passa-faixa gaussiano no domínio da frequência
         - remove BAIXAS frequências  -> fundo do céu, Via Láctea, nebulosas
         - remove ALTAS frequências   -> ruído de sensor pixel a pixel
    3. Segmentação: limiarização (Otsu ou k·sigma) + abertura morfológica
    4. Rotulagem de componentes conexos -> centroide, área e fluxo de cada estrela
    5. Decisão: ranking por fluxo -> estrelas principais da constelação
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------- entrada
def carregar(caminho: str):
    bgr = cv2.imread(caminho)
    if bgr is None:
        raise FileNotFoundError(caminho)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cinza = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return rgb, cinza


# ---------------------------------------------------------------- FFT
def _fft_centralizada(img: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))


def espectro_magnitude(img: np.ndarray) -> np.ndarray:
    """log(1 + |F(u,v)|), com a frequência zero (DC) no centro."""
    return np.log1p(np.abs(_fft_centralizada(img)))


def espectro_fase(img: np.ndarray) -> np.ndarray:
    return np.angle(_fft_centralizada(img))


def mascara_gaussiana(shape, tipo: str, d0: float, d1: float | None = None) -> np.ndarray:
    """H(u,v) gaussiana. tipo: 'baixa', 'alta' ou 'banda' (d0 = corte inferior, d1 = superior)."""
    h, w = shape
    v, u = np.ogrid[:h, :w]
    d2 = (u - w // 2) ** 2 + (v - h // 2) ** 2
    baixa = lambda d: np.exp(-d2 / (2.0 * d ** 2))
    if tipo == "baixa":
        return baixa(d0)
    if tipo == "alta":
        return 1.0 - baixa(d0)
    if tipo == "banda":
        return baixa(d1) * (1.0 - baixa(d0))
    raise ValueError(tipo)


def filtro_fft(img: np.ndarray, tipo: str, d0: float, d1: float | None = None,
               pad: int | None = None) -> np.ndarray:
    """Filtra no domínio da frequência: IFFT( FFT(img) * H ).

    A imagem é espelhada nas bordas antes da FFT: a FFT supõe que a imagem
    se repete periodicamente, e um gradiente de fundo cria um 'degrau' na
    emenda que vira vazamento espectral (artefatos nas bordas).
    """
    pad = max(img.shape) // 8 if pad is None else pad
    x = cv2.copyMakeBorder(img.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_REFLECT)
    H = mascara_gaussiana(x.shape, tipo, d0, d1)
    y = np.real(np.fft.ifft2(np.fft.ifftshift(_fft_centralizada(x) * H)))
    return y[pad:-pad, pad:-pad]


def cortes_padrao(shape) -> tuple[float, float]:
    """Cortes em função do tamanho N da imagem (em 'ciclos por imagem').
    Fundo/nebulosas: estruturas maiores que ~N/4 px  -> d < N/100 ~ remove.
    Ruído: estruturas de 1 px -> d > N/6 ~ remove. Estrelas (PSF 1.5-4 px) ficam no meio."""
    n = max(shape)
    return n / 100.0, n / 6.0


# ---------------------------------------------------------------- segmentação
def limiar(img: np.ndarray, metodo: str = "otsu", k: float = 5.0) -> tuple[np.ndarray, float]:
    """Binariza. 'otsu' = minimiza variância intra-classe do histograma.
    'sigma' = mediana + k·σ robusto (MAD), padrão em astronomia (SExtractor)."""
    x = img.astype(np.float32)
    if metodo == "otsu":
        x8 = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        t8, b = cv2.threshold(x8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = x.min() + t8 / 255.0 * (x.max() - x.min())
        return b, float(t)
    if metodo == "sigma":
        med = np.median(x)
        sigma = 1.4826 * np.median(np.abs(x - med))
        t = med + k * sigma
        return ((x > t) * 255).astype(np.uint8), float(t)
    raise ValueError(metodo)


def componentes(binaria: np.ndarray, intensidade: np.ndarray, area_min: int = 3) -> list[dict]:
    """Rotula regiões conexas (8-vizinhança) e mede cada estrela."""
    n, rot, stats, cent = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    estrelas = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_min:
            continue
        mascara = rot == i
        estrelas.append({
            "x": float(cent[i, 0]), "y": float(cent[i, 1]),
            "area": int(area),
            "fluxo": float(np.clip(intensidade[mascara], 0, None).sum()),
        })
    return sorted(estrelas, key=lambda e: e["fluxo"], reverse=True)


def segmentar_estrelas(cinza: np.ndarray, metodo: str = "otsu", area_min: int = 3) -> list[tuple]:
    """Linha de base (pipeline original do grupo): Gaussiano 5x5 + Otsu, sem FFT."""
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)
    b, _ = limiar(suave, metodo)
    return [(e["x"], e["y"]) for e in componentes(b, suave.astype(np.float32), area_min)]


def pipeline(cinza: np.ndarray, d0: float | None = None, d1: float | None = None,
             metodo: str = "sigma", k: float = 5.0, area_min: int = 3) -> dict:
    """Pipeline completo. Retorna todas as etapas intermediárias para visualização."""
    pd0, pd1 = cortes_padrao(cinza.shape)
    d0, d1 = d0 or pd0, d1 or pd1
    filtrada = filtro_fft(cinza, "banda", d0, d1)
    binaria, t = limiar(filtrada, metodo, k)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    comps = componentes(binaria, filtrada, area_min)
    return {
        "filtrada": filtrada, "binaria": binaria, "limiar": t, "d0": d0, "d1": d1,
        "componentes": comps, "estrelas": [(e["x"], e["y"]) for e in comps],
    }


# ---------------------------------------------------------------- validação
def campo_sintetico(n_estrelas: int = 50, tamanho: int = 512, gradiente: float = 0.0,
                    ruido: float = 5.0, seed: int = 0):
    """Céu artificial com posições conhecidas (ground truth).
    Estrelas = PSF gaussiana; fundo = rampa linear + 'nebulosa' gaussiana grande; + ruído."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:tamanho, :tamanho].astype(np.float32)
    img = 20 + gradiente * xx / tamanho
    img += 0.6 * gradiente * np.exp(-((xx - 0.3 * tamanho) ** 2 + (yy - 0.6 * tamanho) ** 2)
                                    / (2 * (tamanho / 6) ** 2))
    estrelas = []
    m = 12
    while len(estrelas) < n_estrelas:
        x, y = rng.uniform(m, tamanho - m, 2)
        if any((x - a) ** 2 + (y - b) ** 2 < 15 ** 2 for a, b in estrelas):
            continue
        s, amp = rng.uniform(1.2, 2.5), rng.uniform(60, 200)
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s ** 2))
        estrelas.append((float(x), float(y)))
    img += rng.normal(0, ruido, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), estrelas


def avaliar_deteccao(deteccoes, verdade, tol: float = 4.0) -> dict:
    """Casa detecções com a verdade (distância <= tol px). Precisão, recall, F1."""
    livres = list(verdade)
    vp = 0
    for dx, dy in deteccoes:
        if not livres:
            break
        d = [(dx - x) ** 2 + (dy - y) ** 2 for x, y in livres]
        j = int(np.argmin(d))
        if d[j] <= tol ** 2:
            vp += 1
            livres.pop(j)
    fp, fn = len(deteccoes) - vp, len(verdade) - vp
    p = vp / (vp + fp) if vp + fp else 0.0
    r = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precisao": p, "recall": r, "f1": f1, "vp": vp, "fp": fp, "fn": fn}
