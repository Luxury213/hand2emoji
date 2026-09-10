"""
DETECTOR DE GESTOS EN TIEMPO REAL - HAND2EMOJI (FASE 5)
- Renderizado de emojis Unicode de alta definición con Pillow (PIL) y fuentes del sistema.
- Emisión de video a cámara virtual (OBS Virtual Camera / Zoom / Meet) con pyvirtualcam.
- Inyección automática de texto en el sistema operativo (macro teclado) con pyautogui y pyperclip.
- Normalización canónica y suavizado temporal de predicciones.
"""

import os
import sys
import time
import pickle
import argparse
from collections import deque, Counter
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

import gestures_common as gc

# Reconfigurar salida estándar para evitar errores con caracteres Unicode en Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Importaciones opcionales para macro y cámara virtual
try:
    import pyperclip
except ImportError:
    pyperclip = None

try:
    import pyautogui
    pyautogui.FAILSAFE = True
except ImportError:
    pyautogui = None

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODELS_DIR        = 'models'
CONFIANZA_MINIMA  = 0.70  # Umbral mínimo de confianza para aceptar detección
SUAVIZADO         = 7     # Cantidad de frames para el suavizado temporal
MAX_MANOS         = 2

# ============================================================
# COLORES (BGR)
# ============================================================
COLOR_FONDO    = (15, 15, 18)
COLOR_VERDE    = (100, 220, 52)
COLOR_AMARILLO = (36, 191, 251)
COLOR_ROJO     = (68, 68, 239)
COLOR_BLANCO   = (255, 255, 255)
COLOR_GRIS     = (160, 160, 160)
COLOR_OSCURO   = (24, 24, 28)


class EmojiRenderer:
    """
    Renderiza pictogramas y emojis Unicode en alta definición y color sobre
    imágenes OpenCV (formato NumPy BGR) utilizando Pillow (PIL).
    """

    def __init__(self, font_size=74):
        self.font_size = font_size
        self.font = None
        self.font_path = None

        # Prioridad de fuentes de color según plataforma
        candidatos = [
            "seguiemj.ttf",          # Windows 10/11 Segoe UI Emoji
            "NotoColorEmoji.ttf",     # Linux / Android
            "Apple Color Emoji.ttc",  # macOS
            "DejaVuSans.ttf",         # Linux Fallback
            "arial.ttf"
        ]

        for font_candidate in candidatos:
            try:
                self.font = ImageFont.truetype(font_candidate, self.font_size)
                self.font_path = font_candidate
                break
            except Exception:
                continue

        if self.font is None:
            self.font = ImageFont.load_default()

    def dibujar_emoji_centrado(self, frame_bgr, emoji_str, centro_x, centro_y):
        """
        Dibuja un emoji Unicode perfectamente centrado en las coordenadas (centro_x, centro_y).
        """
        if not emoji_str:
            return frame_bgr

        try:
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            # Bounding box para centrado exacto
            bbox = draw.textbbox((0, 0), emoji_str, font=self.font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            pos_x = int(centro_x - (w // 2) - bbox[0])
            pos_y = int(centro_y - (h // 2) - bbox[1])

            draw.text((pos_x, pos_y), emoji_str, font=self.font, embedded_color=True)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return frame_bgr

    def dibujar_emoji_posicion(self, frame_bgr, emoji_str, pos_xy, font_size=None):
        """
        Dibuja un emoji Unicode en una posición (x, y) específica con tamaño opcional.
        """
        if not emoji_str:
            return frame_bgr

        try:
            font = self.font
            if font_size and self.font_path and font_size != self.font_size:
                try:
                    font = ImageFont.truetype(self.font_path, font_size)
                except Exception:
                    font = self.font

            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            draw.text(pos_xy, emoji_str, font=font, embedded_color=True)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return frame_bgr


class MacroManager:
    """
    Gestiona el copiado al portapapeles y la inyección automática de emojis
    en la aplicación activa del sistema mediante pulsaciones de teclado simuladas.
    """

    def __init__(self, activo=False, hold_frames=20, cooldown_frames=30):
        self.activo = activo
        self.hold_frames = hold_frames
        self.cooldown_frames = cooldown_frames
        self.contador_hold = 0
        self.contador_cooldown = 0
        self.ultimo_gesto = None

    def alternar(self):
        self.activo = not self.activo
        self.contador_hold = 0
        return self.activo

    def procesar_gesto(self, gesto, emoji):
        """
        Evalúa si el gesto cumple el tiempo de retención para disparar la macro.
        Retorna True únicamente en el frame en que se inyecta el emoji.
        """
        if self.contador_cooldown > 0:
            self.contador_cooldown -= 1
            return False

        if not self.activo or not gesto or not emoji:
            self.contador_hold = 0
            self.ultimo_gesto = None
            return False

        if gesto == self.ultimo_gesto:
            self.contador_hold += 1
            if self.contador_hold >= self.hold_frames:
                disparado = self.inyectar_emoji(emoji)
                self.contador_hold = 0
                self.contador_cooldown = self.cooldown_frames
                return disparado
        else:
            self.ultimo_gesto = gesto
            self.contador_hold = 1

        return False

    def porcentaje_progreso(self):
        if not self.activo or self.hold_frames <= 0:
            return 0.0
        return min(1.0, self.contador_hold / float(self.hold_frames))

    @staticmethod
    def copiar_al_portapapeles(texto):
        if not pyperclip:
            return False
        try:
            pyperclip.copy(texto)
            return True
        except Exception:
            return False

    @classmethod
    def inyectar_emoji(cls, emoji):
        copiado = cls.copiar_al_portapapeles(emoji)
        if not copiado:
            return False
        if not pyautogui:
            return True  # Al menos se copió al portapapeles
        try:
            pyautogui.FAILSAFE = False
            pyautogui.hotkey('ctrl', 'v')
            return True
        except Exception:
            # Aunque la simulación de teclas falle, el emoji ya quedó copiado en el portapapeles
            return True


class DetectorGestos:
    """
    Detector de gestos en tiempo real para escritorio con soporte de:
    - Renderizado Unicode de emojis con Pillow
    - Inyección de macros de teclado
    - Transmisión a cámaras virtuales
    """

    def __init__(self, macro_default=False):
        print("🤖 Inicializando detector Hand2Emoji...")

        # Cargar modelo y metadatos
        self._cargar_modelo()

        # MediaPipe Hands
        self.mp_hands          = mp.solutions.hands
        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_MANOS,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Buffer para suavizado temporal (Counter de últimas N predicciones)
        self.buffer_predicciones = deque(maxlen=SUAVIZADO)

        # Estados de predicción actual
        self.gesto_actual      = None
        self.emoji_actual      = None
        self.confianza_actual  = 0.0
        self.lado_actual       = None

        # Renderizadores de emojis
        self.emoji_renderer       = EmojiRenderer(font_size=76)
        self.emoji_renderer_small = EmojiRenderer(font_size=28)

        # Gestor de macros y portapapeles
        self.macro_manager = MacroManager(activo=macro_default)

        # Estado de cámara virtual y notificaciones toast
        self.virtualcam_activo = False
        self.toast_msg         = None
        self.toast_expira      = 0.0

        # Estadísticas de sesión
        self.total_predicciones = 0
        self.tiempo_inicio      = time.time()

        print("✅ Detector listo!\n")

    def _cargar_modelo(self):
        try:
            with open(os.path.join(MODELS_DIR, 'modelo.pkl'), 'rb') as f:
                self.modelo = pickle.load(f)

            with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)

            with open(os.path.join(MODELS_DIR, 'labels.pkl'), 'rb') as f:
                self.le = pickle.load(f)

            with open(os.path.join(MODELS_DIR, 'metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)

            self.emoji_map = self.metadata.get('emoji_map', gc.EMOJI_MAP)
            self.gestos    = self.metadata.get('gestos', list(self.le.classes_))

            print(f"   ✅ Modelo cargado | {len(self.gestos)} gestos")
            print(f"   📋 Gestos: {self.gestos}")

        except FileNotFoundError as e:
            print(f"\n❌ No se encontró el modelo: {e}")
            print("   Ejecuta primero entrenador.py")
            raise

    def extraer_caracteristicas(self, landmarks_raw, lado):
        """Extrae características canónicas con espejado para mano izquierda."""
        n_features = self.metadata.get('n_features', 63)
        if n_features == 63:
            return gc.extraer_caracteristicas_canonicas(landmarks_raw, lado)
        return gc.extraer_features_completas(landmarks_raw, lado)

    def predecir(self, features):
        """Predice el gesto y aplica suavizado por votación en el buffer."""
        features_scaled = self.scaler.transform([features])
        proba = self.modelo.predict_proba(features_scaled)[0]
        idx_max = np.argmax(proba)
        confianza = proba[idx_max]

        if confianza >= CONFIANZA_MINIMA:
            gesto = self.le.classes_[idx_max]
            self.buffer_predicciones.append(gesto)
            gesto_suavizado = Counter(self.buffer_predicciones).most_common(1)[0][0]
            return gesto_suavizado, confianza

        return None, confianza

    def mostrar_toast(self, mensaje, duracion=2.0):
        """Muestra un mensaje emergente temporal en la barra inferior."""
        self.toast_msg = mensaje
        self.toast_expira = time.time() + duracion

    def dibujar_emoji_overlay(self, frame, h, w):
        """
        Dibuja el overlay circular con el emoji en color real usando Pillow
        e indicador de progreso si la macro está acumulando hold.
        """
        if not self.gesto_actual:
            return frame

        emoji = self.emoji_actual or ''
        centro_x, centro_y = w - 105, h // 2

        # 1. Fondo circular semitransparente
        overlay = frame.copy()
        cv2.circle(overlay, (centro_x, centro_y), 75, COLOR_FONDO, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # 2. Borde exterior según confianza
        color_borde = COLOR_VERDE if self.confianza_actual > 0.85 else COLOR_AMARILLO
        cv2.circle(frame, (centro_x, centro_y), 75, color_borde, 2)

        # 3. Anillo de progreso si el modo macro está activo y acumulando
        progreso = self.macro_manager.porcentaje_progreso()
        if progreso > 0:
            angulo_fin = int(progreso * 360)
            cv2.ellipse(frame, (centro_x, centro_y), (80, 80), -90, 0, angulo_fin, (255, 255, 255), 4)

        # 4. Renderizado del emoji con Pillow en alta definición
        frame = self.emoji_renderer.dibujar_emoji_centrado(frame, emoji, centro_x, centro_y)

        # 5. Nombre del gesto debajo del círculo con badge oscuro
        nombre_texto = self.gesto_actual.replace('_', ' ')
        texto_w = cv2.getTextSize(nombre_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        badge_x1 = max(0, centro_x - texto_w // 2 - 8)
        badge_x2 = min(w, centro_x + texto_w // 2 + 8)
        badge_y1 = centro_y + 82
        badge_y2 = centro_y + 104

        overlay_badge = frame.copy()
        cv2.rectangle(overlay_badge, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay_badge, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, nombre_texto, (centro_x - texto_w // 2, centro_y + 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_BLANCO, 1, cv2.LINE_AA)

        return frame

    def dibujar_interfaz(self, frame, h, w):
        """Dibuja los paneles de telemetría, atajos y notificaciones toast."""
        # ---- Panel superior izquierdo ----
        if self.gesto_actual:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (340, 95), COLOR_FONDO, -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # Mini emoji en esquina
            frame = self.emoji_renderer_small.dibujar_emoji_posicion(frame, self.emoji_actual, (12, 10))

            # Nombre del gesto
            cv2.putText(frame, self.gesto_actual.replace('_', ' '), (48, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLANCO, 2, cv2.LINE_AA)

            # Barra de confianza
            barra_w = int(270 * self.confianza_actual)
            cv2.rectangle(frame, (48, 48), (318, 58), (45, 45, 50), -1)
            color_barra = COLOR_VERDE if self.confianza_actual > 0.85 else COLOR_AMARILLO
            cv2.rectangle(frame, (48, 48), (48 + barra_w, 58), color_barra, -1)

            lado_str = f"[{self.lado_actual}] " if self.lado_actual else ""
            cv2.putText(frame, f"{lado_str}{int(self.confianza_actual*100)}%", (48, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GRIS, 1, cv2.LINE_AA)

            # Indicadores de estado
            macro_str = "MACRO: ON" if self.macro_manager.activo else "MACRO: OFF"
            macro_col = COLOR_VERDE if self.macro_manager.activo else COLOR_GRIS
            cv2.putText(frame, macro_str, (180, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.4, macro_col, 1, cv2.LINE_AA)

            if self.virtualcam_activo:
                cv2.putText(frame, "VCAM: ON", (270, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 50), 1, cv2.LINE_AA)

        else:
            # Prompt discreto si no hay mano
            overlay = frame.copy()
            cv2.rectangle(overlay, (12, 12), (220, 48), COLOR_FONDO, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, "Muestra tu mano...", (24, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GRIS, 1, cv2.LINE_AA)

        # ---- Panel inferior: atajos y toast ----
        overlay_bot = frame.copy()
        cv2.rectangle(overlay_bot, (0, h - 35), (w, h), (10, 10, 14), -1)
        cv2.addWeighted(overlay_bot, 0.75, frame, 0.25, 0, frame)

        # Notificación Toast activa o atajos de teclado
        if self.toast_msg and time.time() < self.toast_expira:
            cv2.putText(frame, f"✓ {self.toast_msg}", (15, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_VERDE, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "'q' salir | 's' foto | 'r' reset | 'm' macro | 'c' copiar",
                        (15, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_GRIS, 1, cv2.LINE_AA)

        # Conteo de predicciones
        pred_str = f"Pred: {self.total_predicciones}"
        cv2.putText(frame, pred_str, (w - 110, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_GRIS, 1, cv2.LINE_AA)

        return frame

    def procesar_frame(self, frame):
        """
        Procesa un único frame completo:
        1. MediaPipe detection
        2. Extracción canónica
        3. Predicción
        4. Inyección macro si procede
        5. Overlays y renderizado de emojis con Pillow
        Retorna (frame_procesado, metadata_dict).
        """
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = self.hands.process(frame_rgb)

        mano_detectada = False
        lado = None
        gesto = None
        confianza = 0.0

        if resultados.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(resultados.multi_hand_landmarks, resultados.multi_handedness):
                mano_detectada = True
                lado = handedness.classification[0].label
                self.lado_actual = lado

                landmarks_raw = [(lm.x * w, lm.y * h, lm.z) for lm in hand_landmarks.landmark]
                features = self.extraer_caracteristicas(landmarks_raw, lado)
                gesto, confianza = self.predecir(features)

                if gesto:
                    self.gesto_actual = gesto
                    self.emoji_actual = self.emoji_map.get(gesto, '🫱')
                    self.confianza_actual = confianza
                    self.total_predicciones += 1

                    # Procesar macro automática
                    disparado = self.macro_manager.procesar_gesto(self.gesto_actual, self.emoji_actual)
                    if disparado:
                        self.mostrar_toast(f"Pegado: {self.emoji_actual} ({self.gesto_actual})")

                # Conexiones finas estilizadas
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        if not mano_detectada:
            self.buffer_predicciones.clear()
            self.gesto_actual = None
            self.emoji_actual = None
            self.confianza_actual = 0.0
            self.lado_actual = None
            self.macro_manager.contador_hold = 0

        # Dibujar overlays y emoji en alta definición
        frame = self.dibujar_emoji_overlay(frame, h, w)
        frame = self.dibujar_interfaz(frame, h, w)

        info = {
            'mano_detectada': mano_detectada,
            'gesto': self.gesto_actual,
            'emoji': self.emoji_actual,
            'confianza': self.confianza_actual,
            'lado': self.lado_actual,
            'macro_activo': self.macro_manager.activo,
            'virtualcam_activo': self.virtualcam_activo
        }

        return frame, info

    def run(self, cam_id=0, virtualcam=False, macro=False, no_gui=False):
        """Bucle principal de captura y visualización."""
        print("=" * 60)
        print("🎥 DETECTOR HAND2EMOJI - Tiempo Real (Fase 5)")
        print("=" * 60)
        print("  'q' → Salir")
        print("  's' → Guardar screenshot")
        print("  'r' → Resetear predicción")
        print("  'm' → Alternar modo macro (auto-escritura)")
        print("  'c' → Copiar emoji actual al portapapeles")
        print("=" * 60 + "\n")

        self.macro_manager.activo = macro

        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print(f"❌ No se pudo abrir la cámara {cam_id}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 800
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 600

        # Cámara virtual opcional
        vcam = None
        if virtualcam:
            if pyvirtualcam:
                try:
                    vcam = pyvirtualcam.Camera(width=w, height=h, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
                    self.virtualcam_activo = True
                    print(f"✅ Cámara virtual activa: {vcam.device}")
                except Exception as e:
                    print(f"⚠️ No se pudo iniciar la cámara virtual: {e}")
                    vcam = None
            else:
                print("⚠️ Librería pyvirtualcam no instalada.")

        frames = 0
        t0 = time.time()
        screenshot_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frames += 1
                frame = cv2.flip(frame, 1)

                # Procesamiento modular
                frame, _ = self.procesar_frame(frame)

                # Enviar a cámara virtual
                if vcam:
                    vcam.send(frame)
                    vcam.sleep_until_next_frame()

                # FPS y telemetría
                fps = frames / (time.time() - t0 + 1e-6)
                cv2.putText(frame, f"FPS: {int(fps)}", (w - 110, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_GRIS, 1, cv2.LINE_AA)

                if not no_gui:
                    cv2.imshow('Hand2Emoji - Detector', frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break

                    elif key == ord('s'):
                        os.makedirs('screenshots', exist_ok=True)
                        nombre = f"screenshots/gesto_{self.gesto_actual or 'vacio'}_{screenshot_count}.png"
                        cv2.imwrite(nombre, frame)
                        screenshot_count += 1
                        self.mostrar_toast(f"Screenshot: {nombre}")
                        print(f"  📸 Screenshot guardado: {nombre}")

                    elif key == ord('r'):
                        self.buffer_predicciones.clear()
                        self.gesto_actual = None
                        self.emoji_actual = None
                        self.confianza_actual = 0.0
                        self.mostrar_toast("Predicción reseteada")
                        print("  🔄 Predicción reseteada")

                    elif key == ord('m'):
                        nuevo_estado = self.macro_manager.alternar()
                        estado_str = "Activado" if nuevo_estado else "Desactivado"
                        self.mostrar_toast(f"Macro {estado_str}")
                        print(f"  ⚡ Modo Macro: {estado_str}")

                    elif key == ord('c'):
                        if self.emoji_actual:
                            MacroManager.copiar_al_portapapeles(self.emoji_actual)
                            self.mostrar_toast(f"Copiado: {self.emoji_actual}")
                            print(f"  📋 Emoji copiado: {self.emoji_actual}")

        finally:
            cap.release()
            if vcam:
                vcam.close()
            cv2.destroyAllWindows()
            self.hands.close()

            elapsed = time.time() - self.tiempo_inicio
            print(f"\n👋 Sesión terminada ({int(elapsed)}s) | Predicciones: {self.total_predicciones}")


def parse_args():
    parser = argparse.ArgumentParser(description="Hand2Emoji Desktop Detector")
    parser.add_argument("-c", "--cam", type=int, default=0, help="Índice de la cámara web (default: 0)")
    parser.add_argument("-v", "--virtualcam", action="store_true", help="Emitir a cámara virtual (OBS/Zoom/Meet)")
    parser.add_argument("-m", "--macro", action="store_true", help="Activar modo macro de inyección automática de emojis")
    parser.add_argument("--no-gui", action="store_true", help="Ejecutar en modo headless sin ventana gráfica")
    return parser.parse_args()


# ============================================================
if __name__ == "__main__":
    args = parse_args()
    try:
        detector = DetectorGestos(macro_default=args.macro)
        detector.run(cam_id=args.cam, virtualcam=args.virtualcam, macro=args.macro, no_gui=args.no_gui)
    except KeyboardInterrupt:
        print("\n👋 Programa interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise