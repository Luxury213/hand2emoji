"""
DETECTOR DE GESTOS EN TIEMPO REAL - HAND2EMOJI
Carga el modelo entrenado y detecta gestos por cámara
Muestra el emoji correspondiente en pantalla
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import deque

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODELS_DIR        = 'models'
CONFIANZA_MINIMA  = 0.7   # Solo muestra el gesto si el modelo está >70% seguro
SUAVIZADO         = 7     # Cuántos frames usar para suavizar la predicción
MAX_MANOS         = 2

# ============================================================
# COLORES (BGR)
# ============================================================
COLOR_FONDO    = (20, 20, 20)
COLOR_VERDE    = (0, 220, 100)
COLOR_AMARILLO = (0, 220, 220)
COLOR_ROJO     = (0, 80, 255)
COLOR_BLANCO   = (255, 255, 255)
COLOR_GRIS     = (150, 150, 150)


class DetectorGestos:

    def __init__(self):
        print("🤖 Inicializando detector Hand2Emoji...")

        # --- Cargar modelo y utilidades ---
        self._cargar_modelo()

        # --- MediaPipe ---
        self.mp_hands          = mp.solutions.hands
        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_MANOS,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Buffer para suavizar predicciones (evita parpadeo)
        # Guarda las últimas N predicciones y elige la más frecuente
        self.buffer_predicciones = deque(maxlen=SUAVIZADO)

        # Estado actual mostrado en pantalla
        self.gesto_actual   = None
        self.emoji_actual   = None
        self.confianza_actual = 0.0

        # Estadísticas de sesión
        self.total_predicciones = 0
        self.tiempo_inicio      = time.time()

        print("✅ Detector listo!\n")

    # ----------------------------------------------------------
    # CARGA DEL MODELO
    # ----------------------------------------------------------
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

            self.emoji_map = self.metadata['emoji_map']
            self.gestos    = self.metadata['gestos']

            print(f"   ✅ Modelo cargado | {len(self.gestos)} gestos")
            print(f"   📋 Gestos: {self.gestos}")

        except FileNotFoundError as e:
            print(f"\n❌ No se encontró el modelo: {e}")
            print("   Ejecuta primero entrenador.py")
            raise

    # ----------------------------------------------------------
    # EXTRACCIÓN DE CARACTERÍSTICAS (igual que en recolector)
    # ----------------------------------------------------------
    def extraer_caracteristicas(self, landmarks_raw, lado):
        muneca = np.array(landmarks_raw[0])
        ref    = np.array(landmarks_raw[9])
        escala = np.linalg.norm(ref - muneca) + 1e-6

        puntos = np.array(landmarks_raw)
        norm   = (puntos - muneca) / escala

        xs = norm[:, 0].tolist()
        ys = norm[:, 1].tolist()
        zs = norm[:, 2].tolist()

        lado_num = 1 if lado == 'Right' else 0

        return xs + ys + zs + [lado_num]  # 64 features

    # ----------------------------------------------------------
    # PREDICCIÓN SUAVIZADA
    # ----------------------------------------------------------
    def predecir(self, features):
        """
        Predice el gesto y aplica suavizado temporal.
        El suavizado evita que el emoji parpadee entre frames.
        Elige la predicción más frecuente en los últimos N frames.
        """
        features_scaled = self.scaler.transform([features])

        # Probabilidad por clase
        proba = self.modelo.predict_proba(features_scaled)[0]
        idx_max = np.argmax(proba)
        confianza = proba[idx_max]

        if confianza >= CONFIANZA_MINIMA:
            gesto = self.le.classes_[idx_max]
            self.buffer_predicciones.append(gesto)

            # Suavizado: elegir el gesto más frecuente en el buffer
            from collections import Counter
            gesto_suavizado = Counter(self.buffer_predicciones).most_common(1)[0][0]
            return gesto_suavizado, confianza

        return None, confianza

    # ----------------------------------------------------------
    # INTERFAZ VISUAL
    # ----------------------------------------------------------
    def dibujar_interfaz(self, frame, h, w):

        # ---- Panel superior izquierdo: gesto detectado ----
        if self.gesto_actual:
            emoji = self.emoji_actual or '?'
            gesto = self.gesto_actual

            # Fondo semitransparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (320, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # Nombre del gesto
            cv2.putText(frame, gesto, (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_VERDE, 2)

            # Barra de confianza
            barra_w = int(280 * self.confianza_actual)
            cv2.rectangle(frame, (15, 55), (295, 70), (50, 50, 50), -1)
            color_barra = COLOR_VERDE if self.confianza_actual > 0.85 else COLOR_AMARILLO
            cv2.rectangle(frame, (15, 55), (15 + barra_w, 70), color_barra, -1)
            cv2.putText(frame, f"{self.confianza_actual*100:.0f}%", (300, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLANCO, 1)

            # Texto "Confianza"
            cv2.putText(frame, "confianza", (15, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRIS, 1)

        else:
            cv2.putText(frame, "Haz un gesto...", (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GRIS, 2)

        # ---- Panel inferior: instrucciones ----
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "'q' salir  |  's' screenshot  |  'r' resetear",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRIS, 1)

        # ---- FPS y estadísticas (esquina superior derecha) ----
        elapsed = time.time() - self.tiempo_inicio
        fps_text = f"Pred: {self.total_predicciones}"
        cv2.putText(frame, fps_text, (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRIS, 1)

        return frame

  
    # DIBUJAR EMOJI GRANDE EN EL CENTRO
  
    def dibujar_emoji_overlay(self, frame, h, w):
        """
        Muestra el emoji grande en el centro-derecha de la pantalla
        usando texto Unicode grande
        """
        if not self.gesto_actual:
            return frame

        emoji = self.emoji_actual or ''

        # Fondo circular semitransparente
        centro_x, centro_y = w - 100, h // 2
        overlay = frame.copy()
        cv2.circle(overlay, (centro_x, centro_y), 75, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Borde del círculo según confianza
        color_borde = COLOR_VERDE if self.confianza_actual > 0.85 else COLOR_AMARILLO
        cv2.circle(frame, (centro_x, centro_y), 75, color_borde, 3)

        # Nombre del gesto debajo del círculo
        texto_w = cv2.getTextSize(self.gesto_actual,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(frame, self.gesto_actual,
                    (centro_x - texto_w // 2, centro_y + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLANCO, 1)

        return frame

    # ----------------------------------------------------------
    # BUCLE PRINCIPAL
    # ----------------------------------------------------------
    def run(self):
        print("=" * 55)
        print("🎥 DETECTOR HAND2EMOJI - Tiempo Real")
        print("=" * 55)
        print("  'q' → Salir")
        print("  's' → Guardar screenshot")
        print("  'r' → Resetear predicción")
        print("=" * 55 + "\n")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara")
            return

        frames     = 0
        t0         = time.time()
        screenshot_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames += 1
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = self.hands.process(frame_rgb)

            mano_detectada = False

            if resultados.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    resultados.multi_hand_landmarks,
                    resultados.multi_handedness,
                ):
                    mano_detectada = True
                    lado = handedness.classification[0].label

                    # Extraer landmarks
                    landmarks_raw = [
                        (lm.x * w, lm.y * h, lm.z)
                        for lm in hand_landmarks.landmark
                    ]

                    # Extraer features y predecir
                    features = self.extraer_caracteristicas(landmarks_raw, lado)
                    gesto, confianza = self.predecir(features)

                    if gesto:
                        self.gesto_actual    = gesto
                        self.emoji_actual    = self.emoji_map.get(gesto, '🫱')
                        self.confianza_actual = confianza
                        self.total_predicciones += 1

                        # Log en consola
                        print(f"  {self.emoji_actual} {gesto} "
                              f"[{lado}] {confianza*100:.1f}%")

                    # Dibujar landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Si no hay mano, limpiar predicción después de 1 segundo sin detección
            if not mano_detectada:
                self.buffer_predicciones.clear()
                self.gesto_actual    = None
                self.emoji_actual    = None
                self.confianza_actual = 0.0

            # Dibujar interfaz
            frame = self.dibujar_emoji_overlay(frame, h, w)
            frame = self.dibujar_interfaz(frame, h, w)

            # FPS real
            fps = frames / (time.time() - t0 + 1e-6)
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRIS, 1)

            cv2.imshow('Hand2Emoji - Detector', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                # Screenshot
                os.makedirs('screenshots', exist_ok=True)
                nombre = f"screenshots/gesto_{self.gesto_actual}_{screenshot_count}.png"
                cv2.imwrite(nombre, frame)
                screenshot_count += 1
                print(f"  📸 Screenshot guardado: {nombre}")

            elif key == ord('r'):
                # Resetear predicción
                self.buffer_predicciones.clear()
                self.gesto_actual    = None
                self.emoji_actual    = None
                self.confianza_actual = 0.0
                print("  🔄 Predicción reseteada")

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

        # Resumen
        elapsed = time.time() - self.tiempo_inicio
        print(f"\n👋 Sesión terminada")
        print(f"   Tiempo: {int(elapsed)}s")
        print(f"   Predicciones realizadas: {self.total_predicciones}")


# ============================================================
if __name__ == "__main__":
    try:
        DetectorGestos().run()
    except KeyboardInterrupt:
        print("\n👋 Programa interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise