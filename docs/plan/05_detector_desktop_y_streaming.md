# 🎥 Fase 5: Detector de Escritorio Avanzado, Streaming y Cámara Virtual

Esta fase potencia el script de Python de escritorio ([detector.py](file:///c:/Users/USUARIO/Desktop/Proyectos/Hand2Emoji/detector.py)), convirtiéndolo en una herramienta potente para creadores de contenido, streamers y personas en videollamadas.

---

## 🎯 Objetivos de la Fase 5
1. **Renderizado Real de Emojis en Pantalla**: Solucionar la limitación de OpenCV utilizando **Pillow (PIL)** para dibujar emojis reales con color y alta definición en la ventana de video.
2. **Cámara Virtual para Zoom, Meet y OBS**: Integrar `pyvirtualcam` para que Hand2Emoji aparezca como una cámara web seleccionable en cualquier software de videollamadas.
3. **Escritura Automática en el Sistema Operativo (Macro Teclado)**: Inyectar el emoji detectado directamente en la aplicación activa del sistema mediante `pyautogui` o `pyperclip`.

---

## 🔍 Problema Actual: OpenCV no renderiza Emojis
En `detector.py:198`, el método `dibujar_emoji_overlay` tiene un fondo circular y prepara el texto, pero no puede dibujar el emoji porque `cv2.putText` no soporta caracteres UTF-8 complejos ni pictogramas (muestra `???` o cuadrados vacíos).

---

## 🛠️ Implementación Técnica

### 1. Renderizado de Emojis con Pillow (PIL)
Se convierte el frame de OpenCV (formato NumPy BGR) a imagen PIL RGB, se dibuja el emoji con la fuente del sistema y se devuelve a OpenCV:

```python
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

class EmojiRenderer:
    def __init__(self):
        # En Windows: Segoe UI Emoji. En Linux: NotoColorEmoji.ttf
        try:
            self.font = ImageFont.truetype("seguiemj.ttf", 64)
        except Exception:
            self.font = ImageFont.load_default()

    def dibujar_emoji(self, frame_bgr, emoji_str, pos_xy):
        """
        Dibuja un emoji Unicode sobre un frame BGR de OpenCV sin perder rendimiento.
        """
        # BGR -> RGB -> PIL Image
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Dibujar emoji
        draw.text(pos_xy, emoji_str, font=self.font, embedded_color=True)

        # PIL Image -> NumPy -> BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
```

### 2. Soporte de Cámara Virtual (`pyvirtualcam`)
Permite que el video con la detección y los emojis aparezca como una fuente de webcam en Zoom, Meet o Discord:

```python
import pyvirtualcam

def run_con_camara_virtual(detector):
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30

    print("🎥 Iniciando cámara virtual 'Hand2Emoji Cam'...")
    with pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
        print(f"✅ Cámara virtual activa: {vcam.device}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Procesamiento de Hand2Emoji
            frame_procesado = detector.procesar_frame(frame)

            # Enviar frame procesado a Zoom/Meet/OBS
            vcam.send(frame_procesado)
            vcam.sleep_until_next_frame()

            # Mostrar ventana local de previsualización
            cv2.imshow("Hand2Emoji - Monitor", frame_procesado)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
```

### 3. Inyección de Texto en Aplicaciones Activas
Habilitar una opción para que, al mantener un gesto durante más de 1 segundo, se pegue el emoji en el chat o editor abierto:

```python
import pyperclip
import pyautogui

def escribir_emoji_en_sistema(emoji_char):
    """
    Copia el emoji al portapapeles y simula Ctrl+V para pegarlo
    en la ventana activa sin problemas de codificación.
    """
    pyperclip.copy(emoji_char)
    pyautogui.hotkey('ctrl', 'v')
```

---

## 📦 Nuevas Dependencias Opcionales
Para habilitar estas funcionalidades en escritorio, añadir a `requirements-dev.txt`:
```txt
pillow>=10.2.0
pyvirtualcam>=0.11.0
pyperclip>=1.8.2
pyautogui>=0.9.54
```

---

## ✅ Criterios de Aceptación y Verificación
- [x] El emoji aparece dibujado en alta resolución con colores reales en la ventana de OpenCV gracias a `EmojiRenderer` y `Pillow`.
- [x] La cámara virtual (`pyvirtualcam`) emite frames BGR directamente a OBS Virtual Camera, Zoom y Meet con flag `--virtualcam`.
- [x] Al mantener un gesto con el modo macro activo (`--macro` o tecla 'm'), el emoji se inyecta con `pyautogui` y `pyperclip` tras cumplir la ventana de `hold_frames`.
- [x] Arquitectura desacoplada con método `procesar_frame()` verificado mediante pruebas automatizadas en `test_phase5.py`.

