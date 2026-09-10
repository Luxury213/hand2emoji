# 🛠️ Fase 1: Calidad de Código, Dependencias y Módulos Base

Esta fase sienta las bases técnicas del repositorio para eliminar errores de despliegue en entornos Linux/Render, resolver inconsistencias de codificación de caracteres y desacoplar la lógica común duplicada en múltiples scripts.

---

## 🎯 Objetivos de la Fase 1
1. **Corregir la codificación de [requirements.txt](file:///c:/Users/USUARIO/Desktop/Proyectos/Hand2Emoji/requirements.txt)** de `UTF-16LE` a `UTF-8` estándar.
2. **Dividir las dependencias** en `requirements.txt` (mínimo para API / inferencia básica) y `requirements-dev.txt` (OpenCV, MediaPipe, Matplotlib, Seaborn).
3. **Centralizar la lógica compartida** en un nuevo módulo `gestures_common.py`.
4. **Refactorizar scripts** ([recolector.py](file:///c:/Users/USUARIO/Desktop/Proyectos/Hand2Emoji/recolector.py), [detector.py](file:///c:/Users/USUARIO/Desktop/Proyectos/Hand2Emoji/detector.py), [entrenador.py](file:///c:/Users/USUARIO/Desktop/Proyectos/Hand2Emoji/entrenador.py) y [api.py](file:///c:/Users/USUARIO/Desktop/Proyectos/Hand2Emoji/api.py)) para importar la configuración y funciones comunes desde una sola fuente de verdad.

---

## ⚠️ Problemas Detectados y Soluciones

### 1. Codificación UTF-16LE en `requirements.txt`
* **Causa**: Al ejecutar comandos como `pip freeze > requirements.txt` en PowerShell de Windows, se escribe por defecto con BOM en UTF-16LE (`b'\xff\xfe'`).
* **Impacto**: Servidores basados en Debian/Alpine (como Render, Railway o contenedores Docker) fallan con `UnicodeDecodeError` al ejecutar `pip install -r requirements.txt`.
* **Solución**: Reescribir el archivo en `utf-8` sin BOM.

### 2. Dependencias Incompletas
* **Causa**: `requirements.txt` actual solo contiene FastAPI, Scikit-learn y Uvicorn.
* **Impacto**: Si un desarrollador clona el proyecto e intenta correr `recolector.py` o `detector.py`, fallará porque faltan `opencv-python`, `mediapipe`, `matplotlib` y `seaborn`.

### 3. Duplicación de Código
* **Causa**: El diccionario `EMOJI_MAP` y la función `extraer_caracteristicas(landmarks)` están duplicados en 3 archivos distintos.
* **Impacto**: Si se añade o modifica un gesto, hay que editar manualmente 3 archivos de Python y 1 archivo HTML, con alto riesgo de desincronización.

---

## 📝 Plan de Implementación Paso a Paso

### Paso 1: Crear `gestures_common.py`
Crear un archivo en la raíz del proyecto que defina:
- El mapa oficial de gestos a emojis (`EMOJI_MAP`).
- La lista ordenada de clases.
- La función matemática estándar de extracción y normalización de landmarks.

```python
"""
gestures_common.py - Fuente única de verdad para Hand2Emoji
"""
import numpy as np

EMOJI_MAP = {
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

def normalizar_landmarks(landmarks_raw: list, lado: str = 'Right', espejo_zurdo: bool = True) -> list[float]:
    """
    Normaliza 21 landmarks tridimensionales:
    - Centrado en la muñeca (punto 0)
    - Escalado por distancia muñeca -> base dedo medio (punto 9)
    - Opcional: espejo para mano izquierda (convierte a representación canónica derecha)
    Retorna 63 floats (x0..x20, y0..y20, z0..z20)
    """
    puntos = np.array(landmarks_raw, dtype=np.float32)  # (21, 3)
    muneca = puntos[0]
    ref = puntos[9]

    # Distancia euclidiana muñeca - base dedo medio
    escala = np.linalg.norm(ref - muneca) + 1e-6

    # Traslación y escala
    norm = (puntos - muneca) / escala

    # Espejado canónico: si es mano izquierda, invertir eje X para que coincida con mano derecha
    if espejo_zurdo and lado == 'Left':
        norm[:, 0] = -norm[:, 0]

    xs = norm[:, 0].tolist()
    ys = norm[:, 1].tolist()
    zs = norm[:, 2].tolist()

    return xs + ys + zs
```

### Paso 2: Configurar dependencias limpias
* **`requirements.txt`** (Para despliegues de API / inferencia ligera en Render/Railway):
  ```txt
  fastapi>=0.110.0
  uvicorn>=0.28.0
  pydantic>=2.6.0
  scikit-learn>=1.4.0
  numpy>=1.26.0
  joblib>=1.3.0
  ```
* **`requirements-dev.txt`** (Para desarrollo local, cámara y reentrenamiento):
  ```txt
  -r requirements.txt
  opencv-python>=4.9.0
  mediapipe>=0.10.10
  matplotlib>=3.8.0
  seaborn>=0.13.0
  pandas>=2.2.0
  onnx>=1.15.0
  skl2onnx>=1.16.0
  ```

### Paso 3: Refactorizar Scripts Existentes
1. En `recolector.py`: Importar `EMOJI_MAP` y `normalizar_landmarks` desde `gestures_common`.
2. En `entrenador.py`: Importar `EMOJI_MAP` desde `gestures_common`.
3. En `detector.py`: Importar `EMOJI_MAP` y `normalizar_landmarks` desde `gestures_common`.
4. En `api.py`: Importar `EMOJI_MAP` desde `gestures_common` y tipar estrictamente con Pydantic.

---

## ✅ Criterios de Aceptación y Verificación
- [x] `requirements.txt` se lee correctamente en Linux/PowerShell con codificación UTF-8.
- [x] `python -c "import gestures_common; print(len(gestures_common.EMOJI_MAP))"` imprime `18`.
- [x] `detector.py`, `entrenador.py`, `recolector.py` y `api.py` inician sin errores de importación y pasan los tests de integración en `test_phase1.py`.

