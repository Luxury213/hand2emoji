"""
gestures_common.py - Módulo central y fuente única de verdad para Hand2Emoji
Centraliza constantes de gestos, mapeo de emojis y extracción matemática de landmarks.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

# ============================================================
# MAPA DE GESTOS Y EMOJIS
# ============================================================
EMOJI_MAP: Dict[str, str] = {
    'italiano':         '🤌',
    'rock con pulgar':  '🤟',
    'rock':             '🤘',
    'corazon':          '🫶',
    'ok':               '👌',
    'pulgar':           '👍',
    'paz':              '✌️',
    'puno':             '✊',
    'llamame':          '🤙',
    'mano_abierta':     '✋',
    'indice_izquierda': '👈',
    'indice_derecha':   '👉',
    'indice_arriba':    '👆',
    'indice_abajo':     '👇',
    'dedos_cruzados':   '🤞',
    'fuck_you':         '🖕',
    'te_apunto':        '🫵',
    'pinza':            '🤏',
}

# Tecla (código ASCII) -> nombre del gesto (usado en recolector.py)
TECLAS_GESTOS: Dict[int, str] = {
    ord('1'): 'italiano',
    ord('2'): 'rock con pulgar',
    ord('3'): 'rock',
    ord('4'): 'corazon',
    ord('5'): 'ok',
    ord('6'): 'pulgar',
    ord('7'): 'paz',
    ord('8'): 'puno',
    ord('9'): 'llamame',
    ord('0'): 'mano_abierta',
    ord('j'): 'indice_izquierda',
    ord('k'): 'indice_derecha',
    ord('p'): 'indice_arriba',
    ord('o'): 'indice_abajo',
    ord('l'): 'dedos_cruzados',
    ord('f'): 'fuck_you',
    ord('t'): 'te_apunto',
    ord('m'): 'pinza',
}

NUM_LANDMARKS: int = 21
NUM_FEATURES_LANDMARKS: int = 63   # 21 * (x, y, z)
NUM_FEATURES_TOTAL: int = 64       # 63 landmarks + lado_num


def lado_a_num(lado: str) -> int:
    """Codifica 'Right' como 1 y cualquier otro valor ('Left', etc.) como 0."""
    return 1 if lado == 'Right' else 0


def obtener_emoji(gesto: str, default: str = '🫱') -> str:
    """Retorna el emoji correspondiente a un gesto o un valor por defecto."""
    return EMOJI_MAP.get(gesto, default)


def obtener_columnas_features() -> List[str]:
    """Retorna los nombres de las 64 columnas de características del modelo actual."""
    return (
        [f'p{i}_x' for i in range(NUM_LANDMARKS)]
        + [f'p{i}_y' for i in range(NUM_LANDMARKS)]
        + [f'p{i}_z' for i in range(NUM_LANDMARKS)]
        + ['lado_num']
    )


def extraer_caracteristicas_landmarks(landmarks_raw: List[Tuple[float, float, float]]) -> List[float]:
    """
    Convierte 21 landmarks tridimensionales en 63 valores normalizados:
      - Traslación: resta posición de la muñeca (punto 0)
      - Escala: divide por distancia muñeca -> base dedo medio (punto 9)
        (Invariante a la distancia a la cámara)

    Retorna una lista de 63 floats: [x0..x20, y0..y20, z0..z20]
    """
    if len(landmarks_raw) != NUM_LANDMARKS:
        raise ValueError(f"Se esperaban {NUM_LANDMARKS} landmarks, se recibieron {len(landmarks_raw)}")

    puntos = np.array(landmarks_raw, dtype=np.float32)
    muneca = puntos[0]   # (x, y, z)
    ref    = puntos[9]   # base dedo medio

    escala = float(np.linalg.norm(ref - muneca)) + 1e-6

    norm = (puntos - muneca) / escala

    xs = norm[:, 0].tolist()
    ys = norm[:, 1].tolist()
    zs = norm[:, 2].tolist()

    return xs + ys + zs


def extraer_features_completas(landmarks_raw: List[Tuple[float, float, float]], lado: str) -> List[float]:
    """
    Retorna el vector completo de 64 features (63 landmarks normalizados + lado_num)
    esperado por el modelo legacy.
    """
    return extraer_caracteristicas_landmarks(landmarks_raw) + [lado_a_num(lado)]


def obtener_columnas_canonicas() -> List[str]:
    """Retorna los nombres de las 63 columnas de características canónicas (sin lado_num)."""
    return (
        [f'p{i}_x' for i in range(NUM_LANDMARKS)]
        + [f'p{i}_y' for i in range(NUM_LANDMARKS)]
        + [f'p{i}_z' for i in range(NUM_LANDMARKS)]
    )


def extraer_caracteristicas_canonicas(
    landmarks_raw: List[Tuple[float, float, float]],
    lado: str,
    espejo_zurdo: bool = True
) -> List[float]:
    """
    Convierte 21 landmarks tridimensionales en 63 valores canónicos:
      - Traslación respecto a la muñeca.
      - Escala invariante por distancia muñeca -> base dedo medio.
      - Espejado canónico: si es mano izquierda, invierte eje X para coincidir con la mano derecha.
    Retorna lista de 63 floats (invariante a si es mano izquierda o derecha).
    """
    if len(landmarks_raw) != NUM_LANDMARKS:
        raise ValueError(f"Se esperaban {NUM_LANDMARKS} landmarks, se recibieron {len(landmarks_raw)}")

    puntos = np.array(landmarks_raw, dtype=np.float32)
    muneca = puntos[0]
    ref    = puntos[9]

    escala = float(np.linalg.norm(ref - muneca)) + 1e-6
    norm = (puntos - muneca) / escala

    if espejo_zurdo and lado == 'Left':
        norm[:, 0] = -norm[:, 0]

    xs = norm[:, 0].tolist()
    ys = norm[:, 1].tolist()
    zs = norm[:, 2].tolist()

    return xs + ys + zs


def canonizar_vector_63(landmarks_63: List[float], lado: str) -> List[float]:
    """
    Dado un vector de 63 landmarks ya normalizados (xs + ys + zs),
    invierte las primeras 21 posiciones (coordenadas X) si lado == 'Left'.
    Permite a la API procesar entradas normalizadas de clientes web antiguos o nuevos.
    """
    if lado != 'Left':
        return list(landmarks_63)
    vec = list(landmarks_63)
    for i in range(NUM_LANDMARKS):
        vec[i] = -vec[i]
    return vec


