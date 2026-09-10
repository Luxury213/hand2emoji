# Hand2Emoji

Reconocimiento de gestos de mano en tiempo real que convierte tus movimientos en emojis, corriendo completamente en el navegador.

**[Demo en vivo](https://luxury213.github.io/hand2emoji)** — sin instalación.

---

## Cómo funciona

MediaPipe corre del lado del cliente y extrae 21 landmarks de la mano por frame. Esos landmarks se normalizan con **espejado canónico** (para zurdos y diestros) y se infieren al instante directamente en el propio navegador con **ONNX Runtime Web** (WebAssembly), logrando 60 FPS fluidos y latencia de ~1-3 ms sin enviar video a ningún servidor externo. La aplicación web funciona 100% on-device ($0 de hosting permanente en GitHub Pages) y mantiene fallback automático a la API en FastAPI si se requiere.

---

## Capturas

![demo](assets/screenshot1.jpeg)
![demo](assets/screenshot2.jpeg)
![demo](assets/screenshot3.jpeg)

---

## Stack

| Componente | Tecnología |
|---|---|
| Detección de mano | MediaPipe Hands (client-side) |
| Inferencia Web | ONNX Runtime Web (WebAssembly on-device, 28 KB) |
| Frontend Web | HTML5 / CSS3 Minimalista (Plus Jakarta Sans, Web Audio API) |
| Detector Desktop | OpenCV + Pillow (emojis Unicode en color real) + pyvirtualcam + pyautogui |
| Backend (opcional/fallback) | FastAPI + Uvicorn |
| Machine Learning | Scikit-learn + ONNX + Data Augmentation sintético |


---

## Entrenamiento del modelo

No hay dataset externo — todos los datos fueron recolectados manualmente por mi. Con `recolector.py` grabé 150 muestras por gesto con mis propias manos frente a la webcam. MediaPipe extrae los 21 landmarks por frame, se normalizan y se guardan en un CSV. Ese proceso lo repetí para los 17 gestos.

El entrenamiento corre con `entrenador.py`, que lee el CSV, ajusta un scaler y un clasificador, y guarda cuatro archivos en `models/`: el clasificador, el scaler, el label encoder y un archivo de metadata con el mapa de gestos a emojis.

```bash
python recolector.py   # grabar 150 muestras por gesto
python entrenador.py   # entrenar y guardar modelo
```

Que los datos vengan de una sola persona significa que el modelo es intencionalmente personal — fue entrenado con mis proporciones de mano y mis condiciones de iluminación, lo que afecta qué tan bien generaliza a otros usuarios. Es un tradeoff conocido y un área interesante para mejorar.

---


## Estructura del proyecto

```
hand2emoji/
├── api.py                  # API REST (FastAPI + fallback)
├── detector.py             # Detector de escritorio (Pillow + pyvirtualcam + macros)
├── gestures_common.py      # Módulo común de normalización canónica y constantes
├── recolector.py           # Recolección de datos con MediaPipe
├── entrenador.py           # Entrenamiento, Data Augmentation y exportación ONNX
├── requirements.txt        # Dependencias de producción (UTF-8 sin BOM)
├── requirements-dev.txt    # Dependencias completas de desarrollo
├── test_phase*.py          # Suites de pruebas automatizadas (Fases 1 a 5)
├── models/
│   ├── modelo.onnx         # Modelo ligero optimizado (28 KB)
│   ├── labels.json         # Metadatos para cliente web
│   ├── modelo.pkl          # Modelo Pickle
│   └── scaler.pkl
└── docs/                   # GitHub Pages (Zero Backend)
    ├── index.html          # Web App minimalista con Emoji Composer y Audio
    ├── PLAN_DE_MEJORAS.md  # Hoja de ruta de 5 fases implementadas
    ├── models/             # Activos ONNX servidos estáticamente
    └── plan/               # Especificación técnica detallada por fase
```

## Qué aprendí

La decisión más importante del proyecto no fue técnica sino conceptual: en vez de trabajar con imágenes, enfoqué todo en landmarks de mano. Eso simplificó enormemente el pipeline no necesitaba procesar píxeles, solo coordenadas y me llevó a investigar a fondo cómo funciona MediaPipe por dentro.

También aprendí bastante sobre recolección de datos. No es solo grabar muestras — es grabar muestras útiles. Recolecté cada gesto de frente, de perfil, con distintas iluminaciones y haciendo cada emoji de formas ligeramente diferentes. Ese proceso me hizo entender que la calidad del dataset importa más que el modelo en sí.

En cuanto al alcance, decidí mantener el foco en manos. Podría extenderse a expresiones faciales en el futuro, pero de momento ese límite le da autenticidad al proyecto y es honesto con lo que el modelo puede hacer bien.

Por último, construir e integrar la API fue nuevo para mí. Entender cómo el frontend y el backend se comunican, cómo se estructura un endpoint y cómo se hace un deploy real fue tanto o más valioso que el modelo en sí.
