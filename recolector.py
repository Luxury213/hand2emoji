"""
RECOLECTOR DE GESTOS - 
MediaPipe 0.10.x + Python 3.12
"""

import cv2
import mediapipe as mp
import csv
import os
import shutil
import numpy as np
from collections import deque
import time


# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
META_MUESTRAS    = 150    # Muestras objetivo por gesto
INTERVALO_GUARDADO = 0.3  # Segundos entre guardados 
CSV_FILE         = 'data/mis_gestos.csv'
MAX_MANOS        = 2


class RecolectorGestos:

    # ----------------------------------------------------------
    # EMOJI MAP CENTRALIZADO  
    # ----------------------------------------------------------
    EMOJI_MAP = {
        'italiano':         '🤌',
        'te_quiero':        '🤟',
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
    }

    # Tecla → nombre del gesto
    GESTOS = {
        ord('1'): 'italiano',
        ord('2'): 'te_quiero',
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
    }

    def __init__(self):
        print("🎓 Inicializando recolector de gestos (versión mejorada)...")

        self.mp_hands         = mp.solutions.hands
        self.mp_drawing       = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_MANOS,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Contadores por gesto
        self.contadores = {nombre: 0 for nombre in self.GESTOS.values()}

        # --- Preparar archivo CSV ---
        os.makedirs('data', exist_ok=True)
        self._inicializar_csv()

        # Buffer para suavizar landmarks
        self.buffer_landmarks = deque(maxlen=3)

        # Control de tiempo
        self.ultimo_guardado   = 0
        self.guardados_sesion  = 0
        self.tiempo_inicio_sesion = time.time()

        print(f"✅ Listo | Meta: {META_MUESTRAS} muestras/gesto | "
              f"Gestos: {len(self.GESTOS)}")

    # ----------------------------------------------------------
    # INICIALIZACIÓN DEL CSV
    # ----------------------------------------------------------
    def _inicializar_csv(self):
        if os.path.exists(CSV_FILE):
            # Backup para no perder datos accidentalmente
            backup = CSV_FILE.replace('.csv', '_backup.csv')
            shutil.copy2(CSV_FILE, backup)
            print(f"💾 Backup creado: {backup}")
            self._cargar_contadores()
        else:
            with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # NUEVO: incluye coordenada Z (63 features) + lado de mano
                header = (
                    ['gesto', 'lado']
                    + [f'p{i}_x' for i in range(21)]
                    + [f'p{i}_y' for i in range(21)]
                    + [f'p{i}_z' for i in range(21)]   # ← NUEVO
                )
                writer.writerow(header)
            print(f"📁 Archivo creado: {CSV_FILE}")

    def _cargar_contadores(self):
        try:
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # saltar cabecera
                for row in reader:
                    if row and row[0] in self.contadores:
                        self.contadores[row[0]] += 1
            print("📊 Contadores cargados desde archivo existente")
        except Exception as e:
            print(f"⚠️  No se pudieron cargar contadores: {e}")

    # ----------------------------------------------------------
    # EXTRACCIÓN DE CARACTERÍSTICAS (MEJORADA)
    # ----------------------------------------------------------
    def extraer_caracteristicas(self, landmarks_raw):
        """
        Convierte los 21 puntos MediaPipe en 63 valores normalizados.

        Normalización:
          - Traslación: resta posición de la muñeca (punto 0)
          - Escala:     divide por distancia muñeca→base_dedo_medio (punto 9)
                        → invariante a distancia de la cámara

        Retorna lista de 63 floats: [x0..x20, y0..y20, z0..z20]
        """
        muneca = np.array(landmarks_raw[0])   # (x, y, z)
        ref    = np.array(landmarks_raw[9])   # base dedo medio

        escala = np.linalg.norm(ref - muneca) + 1e-6  # evitar división 0

        puntos = np.array(landmarks_raw)      # (21, 3)
        norm   = (puntos - muneca) / escala   # (21, 3)

        xs = norm[:, 0].tolist()
        ys = norm[:, 1].tolist()
        zs = norm[:, 2].tolist()

        return xs + ys + zs   # 63 valores

    # ----------------------------------------------------------
    # INTERFAZ VISUAL (MEJORADA)
    # ----------------------------------------------------------
    def dibujar_interfaz(self, frame, mano_detectada, lado, h, w):

        # ---- Estado de detección ----
        if mano_detectada:
            label = f"✅ MANO: {lado}"
            color = (0, 255, 0)
        else:
            label = "❌ SIN MANO"
            color = (0, 0, 255)
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ---- Título ----
        cv2.putText(frame, f"GESTOS (meta: {META_MUESTRAS})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # ---- Contadores con barra de progreso ----
        items = sorted(self.contadores.items())
        y_base = 85
        barra_max = 80   # píxeles de ancho máximo para la barra

        for i, (nombre, count) in enumerate(items):
            # tecla correspondiente
            tecla = next(
                (chr(k) for k, v in self.GESTOS.items() if v == nombre), '?'
            )
            emoji = self.EMOJI_MAP.get(nombre, '🫱')
            progreso = min(count / META_MUESTRAS, 1.0)
            color_texto = (0, 255, 0) if count >= META_MUESTRAS else (0, 255, 255)

            col = 0 if i < 8 else 1
            fila = i if i < 8 else i - 8
            x = 10 + col * 230
            y = y_base + fila * 22

            # Texto
            cv2.putText(
                frame,
                f"{tecla}: {nombre} {count}/{META_MUESTRAS}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_texto, 1,
            )

            # Barra de progreso
            bx = x
            by = y + 3
            cv2.rectangle(frame, (bx, by), (bx + barra_max, by + 5),
                          (60, 60, 60), -1)
            cv2.rectangle(frame, (bx, by),
                          (bx + int(barra_max * progreso), by + 5),
                          (0, 200, 100) if progreso >= 1 else (0, 180, 255), -1)

        # ---- Estadísticas de sesión ----
        elapsed = int(time.time() - self.tiempo_inicio_sesion)
        cv2.putText(
            frame,
            f"Sesion: {self.guardados_sesion} guardados | {elapsed}s",
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # ---- Instrucción inferior ----
        cv2.putText(
            frame,
            "Manten tecla para guardar | 'q' para salir",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )

        return frame

    # ----------------------------------------------------------
    # BUCLE PRINCIPAL
    # ----------------------------------------------------------
    def run(self):
        print("\n" + "=" * 65)
        print("🎓  RECOLECTOR MEJORADO - 15 GESTOS")
        print("=" * 65)
        print(f"\n  Meta por gesto : {META_MUESTRAS} muestras")
        print(f"  Features/fila  : 63  (x, y, z normalizados) + lado\n")
        print("  TECLAS:")
        for tecla_ord, nombre in sorted(self.GESTOS.items()):
            emoji = self.EMOJI_MAP.get(nombre, '🫱')
            print(f"    '{chr(tecla_ord)}' → {emoji}  {nombre}")
        print("\n  'q' → Salir")
        print("=" * 65 + "\n")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ ERROR: No se pudo abrir la cámara")
            return

        frames = 0
        t0 = time.time()
        lado_detectado = "N/A"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames += 1
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = self.hands.process(frame_rgb)

            mano_detectada = False

            if resultados.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    resultados.multi_hand_landmarks,
                    resultados.multi_handedness,       # ← NUEVO: lado
                ):
                    mano_detectada = True
                    lado_detectado = handedness.classification[0].label  # 'Left'/'Right'

                    # Extraer landmarks crudos (x, y en píxeles, z normalizado)
                    landmarks_raw = []
                    for lm in hand_landmarks.landmark:
                        landmarks_raw.append((lm.x * w, lm.y * h, lm.z))

                    self.buffer_landmarks.append((landmarks_raw, lado_detectado))

                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

            frame = self.dibujar_interfaz(
                frame, mano_detectada, lado_detectado, h, w
            )

            # FPS
            fps = frames / (time.time() - t0 + 1e-6)
            cv2.putText(frame, f"FPS:{int(fps)}", (w - 90, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('RECOLECTOR - 15 Gestos', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # ---- Guardar muestra ----
            ahora = time.time()
            if (key in self.GESTOS
                    and self.buffer_landmarks
                    and ahora - self.ultimo_guardado > INTERVALO_GUARDADO):

                nombre_gesto = self.GESTOS[key]
                landmarks_raw, lado = self.buffer_landmarks[-1]
                features = self.extraer_caracteristicas(landmarks_raw)

                try:
                    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([nombre_gesto, lado] + features)

                    self.contadores[nombre_gesto] += 1
                    self.guardados_sesion += 1
                    self.ultimo_guardado = ahora

                    emoji = self.EMOJI_MAP.get(nombre_gesto, '🫱')
                    total = self.contadores[nombre_gesto]
                    print(f"  ✅ {emoji} {nombre_gesto} [{lado}] "
                          f"({total}/{META_MUESTRAS})")

                    if total == META_MUESTRAS:
                        print(f"  🎉 ¡Completaste {emoji} {nombre_gesto}!")

                except Exception as e:
                    print(f"  ❌ Error al guardar: {e}")

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self._imprimir_resumen(time.time() - t0)

    # ----------------------------------------------------------
    # RESUMEN FINAL
    # ----------------------------------------------------------
    def _imprimir_resumen(self, elapsed):
        print("\n" + "=" * 65)
        print("📊  RESUMEN FINAL DE RECOLECCIÓN")
        print("=" * 65)
        print(f"  ⏱️  Tiempo       : {int(elapsed)}s")
        print(f"  💾  Guardados   : {self.guardados_sesion} en esta sesión")
        print(f"  📁  Archivo     : {CSV_FILE}\n")

        todos_ok = True
        for nombre, count in sorted(self.contadores.items()):
            emoji = self.EMOJI_MAP.get(nombre, '🫱')
            if count >= META_MUESTRAS:
                print(f"  {emoji}  {nombre}: {count}/{META_MUESTRAS} ✅")
            else:
                faltan = META_MUESTRAS - count
                print(f"  {emoji}  {nombre}: {count}/{META_MUESTRAS} ❌  (faltan {faltan})")
                todos_ok = False

        print("=" * 65)
        if todos_ok:
            print("🎉 ¡Todos los gestos completados! → ejecuta entrenador.py")
        else:
            print("💪 Sigue recolectando los gestos que faltan")
        print("=" * 65 + "\n")


# ============================================================
if __name__ == "__main__":
    try:
        RecolectorGestos().run()
    except KeyboardInterrupt:
        print("\n👋 Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        raise