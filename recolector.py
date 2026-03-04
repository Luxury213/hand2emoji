"""
RECOLECTOR DE GESTOS - VERSIÓN DEFINITIVA CON 14 GESTOS
MediaPipe 0.10.21 + Python 3.12
Presiona números y símbolos para guardar diferentes gestos
"""

import cv2
import mediapipe as mp
import csv
import os
from collections import deque
import time

class RecolectorGestos:
    def __init__(self):
        print("🎓 Inicializando recolector de gestos...")
        
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configurar detector de manos
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # GESTOS A RECOLECTAR - 14 GESTOS POPULARES
        
        self.gestos = {
            # Gestos originales (10)
            ord('1'): 'italiano',           # 🤌
            ord('2'): 'te_quiero',          # 🤟
            ord('3'): 'rock',               # 🤘
            ord('4'): 'corazon',            # 🫶
            ord('5'): 'ok',                 # 👌
            ord('6'): 'pulgar',             # 👍
            ord('7'): 'paz',                # ✌️
            ord('8'): 'spiderman',          # 🫰
            ord('9'): 'llamame',            # 🤙
            ord('0'): 'mano_abierta',       # ✋
            ord('j'): 'indice_izquierda',   # 👈
            ord('k'): 'indice_derecha',     # 👉
            ord('p'): 'indice_arriba',      # 👆
            ord('o'): 'indice_abajo',       # 👇
            ord('l'): 'dedos_cruzados',     # 🤞
        }
        
        # Contadores
        self.contadores = {nombre: 0 for nombre in self.gestos.values()}
        
        # Archivo donde se guardarán los datos
        self.csv_file = 'data/mis_gestos.csv'
        
        # Crear carpeta data si no existe
        os.makedirs('data', exist_ok=True)
        
        # Crear el archivo CSV si no existe
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Cabecera: gesto + 42 números (21 puntos x 2 coordenadas)
                header = ['gesto'] + [f'p{i}_x' for i in range(21)] + [f'p{i}_y' for i in range(21)]
                writer.writerow(header)
                print(f"📁 Archivo creado: {self.csv_file}")
        else:
            # Si el archivo ya existe, leer los contadores actuales
            self.cargar_contadores_existentes()
        
        # Buffer para suavizar (evita guardar movimientos bruscos)
        self.buffer_landmarks = deque(maxlen=3)
        
        # Variables para control de tiempo
        self.ultimo_guardado = 0
        self.intervalo_guardado = 0.5  # 0.5 segundos entre guardados
        
        print("✅ Recolector inicializado correctamente")
        print(f"📊 Total de gestos a recolectar: {len(self.gestos)}")
    
    def cargar_contadores_existentes(self):
        """Carga los contadores desde el archivo CSV existente"""
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Saltar cabecera
                for row in reader:
                    if row and len(row) > 0:  # Si la fila no está vacía
                        gesto = row[0]
                        if gesto in self.contadores:
                            self.contadores[gesto] += 1
            print(f"📊 Contadores cargados desde archivo existente")
        except Exception as e:
            print(f"⚠️ No se pudieron cargar contadores: {e}")
    
    def extraer_caracteristicas(self, landmarks):
        """
        Convierte los 21 puntos de la mano en 42 números
        Normaliza respecto a la muñeca para ser invariante a posición
        """
        caracteristicas = []
        muneca = landmarks[0]  # Punto de referencia (muñeca)
        
        for punto in landmarks:
            # Restar posición de la muñeca para normalizar
            x_norm = punto[0] - muneca[0]
            y_norm = punto[1] - muneca[1]
            caracteristicas.extend([x_norm, y_norm])
        
        return caracteristicas
    
    def dibujar_interfaz(self, frame, mano_detectada, h):
        """Dibuja toda la información en pantalla"""
        
        # Estado de detección
        if mano_detectada:
            cv2.putText(frame, "✅ MANO DETECTADA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "❌ SIN MANO", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Título de contadores
        cv2.putText(frame, "EJEMPLOS GUARDADOS:", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Mostrar contadores de cada gesto
        y_pos = 85
        # Ordenar gestos para mostrarlos de forma organizada
        items_ordenados = sorted(self.contadores.items())
        
        for i, (nombre, count) in enumerate(items_ordenados):
            # Encontrar la tecla correspondiente
            tecla = None
            for k, v in self.gestos.items():
                if v == nombre:
                    tecla = chr(k) if 32 <= k <= 126 else f"\\x{k:x}"
                    break
            
            color = (0, 255, 0) if count >= 30 else (0, 255, 255)
            
            # Mostrar en dos columnas para aprovechar espacio
            if i < 7:  # Primera columna
                x_pos = 20
                y_actual = y_pos + i * 20
            else:  # Segunda columna
                x_pos = 220
                y_actual = y_pos + (i-7) * 20
            
            # Emoji según el gesto (mapeo visual)
            emoji_map = {
                'italiano': '🤌', 'te_quiero': '🤟', 'rock': '🤘',
                'corazon': '🫶', 'ok': '👌', 'pulgar': '👍',
                'paz': '✌️', 'spiderman': '🕸️', 'llamame': '🤙',
                'mano_abierta': '✋', 'indice_izquierda': '👈',
                'indice_derecha': '👉', 'indice_arriba': '👆',
                'indice_abajo': '👇', 'dedos_cruzados': '🤞'
            }
            emoji = emoji_map.get(nombre, '🫱')
            
            cv2.putText(frame, f"{tecla}: {emoji} {nombre} {count}/30", 
                       (x_pos, y_actual),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Instrucción inferior
        cv2.putText(frame, "Presiona tecla para guardar - 'q' para salir", 
                   (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Bucle principal de recolección"""
        print("\n" + "=" * 70)
        print("🎓 RECOLECTOR DE GESTOS - VERSIÓN CON 14 GESTOS")
        print("=" * 70)
        print("\n📝 INSTRUCCIONES:")
        print("1. Haz un gesto con la mano")
        print("2. Cuando veas los puntos verdes, presiona la tecla")
        print("3. Repite 30 veces cada gesto en diferentes posiciones")
        print("\n⌨️ TECLAS Y GESTOS:")
        
        # Mostrar teclas y gestos de forma organizada
        teclas_ordenadas = sorted(self.gestos.keys())
        for i, tecla in enumerate(teclas_ordenadas):
            gesto = self.gestos[tecla]
            tecla_char = chr(tecla) if 32 <= tecla <= 126 else f"\\x{tecla:x}"
            
            # Mapeo de emojis
            emoji_map = {
                'italiano': '🤌', 'te_quiero': '🤟', 'rock': '🤘',
                'corazon': '🫶', 'ok': '👌', 'pulgar': '👍',
                'paz': '✌️', 'spiderman': '🕸️', 'llamame': '🤙',
                'mano_abierta': '✋', 'indice_izquierda': '👈',
                'indice_derecha': '👉', 'indice_arriba': '👆',
                'indice_abajo': '👇', 'dedos_cruzados': '🤞'
            }
            emoji = emoji_map.get(gesto, '🫱')
            
            print(f"   '{tecla_char}' → {emoji} {gesto}")
        
        print("\n   'q' → Salir")
        print("=" * 70)
        
        # Abrir cámara
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # Un poco más grande para ver mejor
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ ERROR: No se pudo abrir la cámara")
            print("   Verifica que ninguna otra aplicación la esté usando")
            return
        
        print("✅ Cámara abierta correctamente")
        print("⏳ Esperando detección de manos...\n")
        
        # Variables para estadísticas
        frames_procesados = 0
        tiempo_inicio = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ ERROR: No se pudo leer el frame")
                break
            
            frames_procesados += 1
            
            # Efecto espejo (más natural)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convertir a RGB (MediaPipe requiere RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            resultados = self.hands.process(frame_rgb)
            
            # Variables para mostrar
            mano_detectada = False
            
            # Si se detectan manos
            if resultados.multi_hand_landmarks:
                for hand_landmarks in resultados.multi_hand_landmarks:
                    mano_detectada = True
                    
                    # Extraer coordenadas de los 21 puntos
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        landmarks.append((x, y))
                    
                    # Guardar en buffer (para evitar movimientos bruscos)
                    self.buffer_landmarks.append(landmarks)
                    
                    # DIBUJAR puntos y conexiones de la mano
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Dibujar interfaz
            frame = self.dibujar_interfaz(frame, mano_detectada, h)
            
            # Mostrar FPS
            tiempo_actual = time.time()
            fps = frames_procesados / (tiempo_actual - tiempo_inicio + 0.001)
            cv2.putText(frame, f"FPS: {int(fps)}", (w-100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow('RECOLECTOR - 14 Gestos', frame)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            
            # Salir con 'q'
            if key == ord('q'):
                break
            
            # Guardar gesto si:
            # 1. Se presionó una tecla válida
            # 2. Hay landmarks en el buffer
            # 3. Pasó suficiente tiempo desde el último guardado
            tiempo_actual = time.time()
            if (key in self.gestos and 
                self.buffer_landmarks and 
                tiempo_actual - self.ultimo_guardado > self.intervalo_guardado):
                
                nombre_gesto = self.gestos[key]
                
                # Usar el último landmarks del buffer
                landmarks = self.buffer_landmarks[-1]
                
                # Extraer características
                caracteristicas = self.extraer_caracteristicas(landmarks)
                
                # Guardar en CSV
                try:
                    with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([nombre_gesto] + caracteristicas)
                    
                    # Actualizar contador
                    self.contadores[nombre_gesto] += 1
                    self.ultimo_guardado = tiempo_actual
                    
                    # Emoji para feedback
                    emoji_map = {
                        'italiano': '🤌', 'te_quiero': '🤟', 'rock': '🤘',
                        'corazon': '🫶', 'ok': '👌', 'pulgar': '👍',
                        'paz': '✌️', 'spiderman': '🕸️', 'llamame': '🤙',
                        'mano_abierta': '✋', 'indice_izquierda': '👈',
                        'indice_derecha': '👉', 'indice_arriba': '👆',
                        'indice_abajo': '👇', 'dedos_cruzados': '🤞'
                    }
                    emoji = emoji_map.get(nombre_gesto, '🫱')
                    
                    # Feedback visual en consola
                    total_gesto = self.contadores[nombre_gesto]
                    print(f"✅ Guardado: {emoji} {nombre_gesto} ({total_gesto}/30)")
                    
                    # Si llegó a 30, mensaje especial
                    if total_gesto == 30:
                        print(f"🎉 ¡Completaste {emoji} {nombre_gesto}! Sigue con otro gesto")
                    
                except Exception as e:
                    print(f"❌ Error al guardar: {e}")
        
        # Cerrar todo
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Resumen final
        print("\n" + "=" * 70)
        print("📊 RESUMEN FINAL DE RECOLECCIÓN")
        print("=" * 70)
        print(f"⏱️  Tiempo total: {int(time.time() - tiempo_inicio)} segundos")
        print(f"📁 Archivo: {self.csv_file}")
        print("\n📈 Progreso por gesto:")
        
        todos_completados = True
        emoji_map = {
            'italiano': '🤌', 'te_quiero': '🤟', 'rock': '🤘',
            'corazon': '🫶', 'ok': '👌', 'pulgar': '👍',
            'paz': '✌️', 'spiderman': '🕸️', 'llamame': '🤙',
            'mano_abierta': '✋', 'indice_izquierda': '👈',
            'indice_derecha': '👉', 'indice_arriba': '👆',
            'indice_abajo': '👇', 'dedos_cruzados': '🤞'
        }
        
        for nombre, count in sorted(self.contadores.items()):
            emoji = emoji_map.get(nombre, '🫱')
            if count >= 30:
                print(f"  {emoji} {nombre}: {count}/30 ✅ COMPLETADO")
            else:
                print(f"  {emoji} {nombre}: {count}/30 ❌ FALTAN {30-count}")
                todos_completados = False
        
        print("=" * 70)
        if todos_completados:
            print("🎉 ¡FELICIDADES! Todos los gestos tienen 30+ ejemplos")
            print("🚀 Ahora puedes entrenar el modelo con 'entrenador.py'")
        else:
            print("💪 Sigue recolectando los gestos que faltan")
            print("   Vuelve a ejecutar el programa cuando quieras continuar")
        print("=" * 70)

if __name__ == "__main__":
    try:
        recolector = RecolectorGestos()
        recolector.run()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("Por favor, copia este error y compártelo para ayudarte")