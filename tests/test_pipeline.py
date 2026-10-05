import numpy as np
import pytest

from src.pipeline import (
    campo_sintetico,
    espectro_magnitude,
    filtro_fft,
    segmentar_estrelas,
    avaliar_deteccao,
    pipeline,
)


def test_espectro_tem_dc_no_centro():
    img = np.full((64, 64), 100, np.uint8)
    mag = espectro_magnitude(img)
    assert np.unravel_index(mag.argmax(), mag.shape) == (32, 32)


def test_passa_baixa_remove_ponto_isolado_e_passa_alta_remove_fundo():
    img = np.zeros((128, 128), np.float32) + 50
    img[64, 64] = 255
    baixa = filtro_fft(img, "baixa", d0=10)
    alta = filtro_fft(img, "alta", d0=10)
    assert baixa[64, 64] < 100          # pico espalhado
    assert abs(alta[10, 10]) < 5        # fundo constante (DC) removido


def test_campo_sintetico_retorna_verdade():
    img, estrelas = campo_sintetico(n_estrelas=30, seed=1)
    assert img.shape == (512, 512)
    assert len(estrelas) == 30


def test_segmentacao_encontra_estrelas_do_campo_sintetico():
    img, verdade = campo_sintetico(n_estrelas=40, seed=2)
    deteccoes = segmentar_estrelas(img)
    m = avaliar_deteccao(deteccoes, verdade)
    assert m["recall"] > 0.8 and m["precisao"] > 0.8


def test_fft_melhora_deteccao_com_gradiente_forte():
    img, verdade = campo_sintetico(n_estrelas=40, seed=3, gradiente=120, ruido=12)
    sem = avaliar_deteccao(segmentar_estrelas(img), verdade)
    com = avaliar_deteccao(pipeline(img)["estrelas"], verdade)
    assert com["f1"] > sem["f1"]


def test_avaliar_deteccao_perfeita():
    pts = [(10, 10), (50, 50)]
    m = avaliar_deteccao(pts, pts)
    assert m == pytest.approx({"precisao": 1, "recall": 1, "f1": 1, "vp": 2, "fp": 0, "fn": 0})
