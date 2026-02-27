"""
Script de verificación
"""
import mediapipe as mp
import cv2
import numpy as np
import sklearn
import pandas as pd

print("=" * 50)
print("VERIFICACIÓN DE INSTALACIÓN")
print("=" * 50)

# Verificar MediaPipe
print(f"MediaPipe versión: {mp.__version__}")

try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    print("✅ mp.solutions.hands importado correctamente")
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7
    )
    print("✅ Objeto Hands creado correctamente")
    hands.close()
except Exception as e:
    print(f"❌ Error: {e}")

# Verificar otras librerías
print(f"✅ OpenCV: {cv2.__version__}")
print(f"✅ NumPy: {np.__version__}")
print(f"✅ scikit-learn: {sklearn.__version__}")
print(f"✅ pandas: {pd.__version__}")

print("=" * 50)