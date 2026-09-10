"""
test_phase1.py - Suite de verificación automatizada para la Fase 1
Verifica codificación de requirements, módulo gestures_common e integración
con api.py, detector.py, entrenador.py y recolector.py.
"""

import os
import sys
import unittest
import numpy as np

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


class TestPhase1(unittest.TestCase):

    def test_requirements_encoding(self):
        """Verifica que requirements.txt sea UTF-8 sin BOM (evita errores en Linux/Render)."""
        self.assertTrue(os.path.exists("requirements.txt"), "requirements.txt debe existir")
        with open("requirements.txt", "rb") as f:
            header = f.read(4)
            self.assertNotEqual(header[:2], b'\xff\xfe', "No debe tener BOM UTF-16LE")
            self.assertNotEqual(header[:2], b'\xfe\xff', "No debe tener BOM UTF-16BE")

        with open("requirements.txt", "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("fastapi", content)
            self.assertIn("scikit-learn", content)

    def test_requirements_dev_exists(self):
        """Verifica que requirements-dev.txt exista y contenga herramientas de desarrollo."""
        self.assertTrue(os.path.exists("requirements-dev.txt"), "requirements-dev.txt debe existir")
        with open("requirements-dev.txt", "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("opencv-python", content)
            self.assertIn("mediapipe", content)

    def test_gestures_common(self):
        """Verifica constantes y funciones matemáticas en gestures_common."""
        import gestures_common as gc

        # Mapeos
        self.assertEqual(len(gc.EMOJI_MAP), 18)
        self.assertEqual(len(gc.TECLAS_GESTOS), 18)
        self.assertEqual(gc.obtener_emoji('italiano'), '🤌')
        self.assertEqual(gc.obtener_emoji('inexistente', '❓'), '❓')

        # Codificación de lado
        self.assertEqual(gc.lado_a_num('Right'), 1)
        self.assertEqual(gc.lado_a_num('Left'), 0)
        self.assertEqual(gc.lado_a_num('Otro'), 0)

        # Dimensiones de características
        cols = gc.obtener_columnas_features()
        self.assertEqual(len(cols), 64)
        self.assertEqual(cols[-1], 'lado_num')

        # Normalización matemática de landmarks
        # Puntos dummy: muñeca en (0,0,0), MCP medio en (0,2,0)
        dummy_landmarks = [(0.0, 0.0, 0.0)] * 21
        dummy_landmarks[9] = (0.0, 2.0, 0.0)  # escala = 2.0
        dummy_landmarks[4] = (1.0, 2.0, 0.0)

        feat63 = gc.extraer_caracteristicas_landmarks(dummy_landmarks)
        self.assertEqual(len(feat63), 63)
        # Punto 4 normalizado: x = 1.0 / 2.0 = 0.5
        self.assertAlmostEqual(feat63[4], 0.5, places=3)

        # Características completas (64 features)
        feat64_der = gc.extraer_features_completas(dummy_landmarks, 'Right')
        self.assertEqual(len(feat64_der), 64)
        self.assertEqual(feat64_der[-1], 1)

        feat64_izq = gc.extraer_features_completas(dummy_landmarks, 'Left')
        self.assertEqual(feat64_izq[-1], 0)

    def test_api_integration(self):
        """Verifica inicialización e inferencia en api.py con gestures_common."""
        import api
        from api import LandmarksInput, predecir

        self.assertGreater(len(api.handler.gestos), 0)
        req = LandmarksInput(landmarks=[0.0] * 63, lado='Right')
        res = predecir(req)

        self.assertIn('gesto', res)
        self.assertIn('emoji', res)
        self.assertIn('confianza', res)
        self.assertIn('top3', res)
        self.assertEqual(len(res['top3']), 3)

    def test_entrenador_integration(self):
        """Verifica que entrenador.py use gestures_common para preprocesamiento."""
        import entrenador as ent
        df = ent.cargar_datos()
        self.assertGreater(len(df), 0)
        X, y, le = ent.preprocesar(df)
        self.assertIn(X.shape[1], [63, 64])
        self.assertEqual(len(X), len(df))

    def test_detector_integration(self):
        """Verifica que detector.py extraiga features mediante gestures_common."""
        import detector
        det = detector.DetectorGestos()
        dummy_lms = [(float(i), float(i), 0.0) for i in range(21)]
        feats = det.extraer_caracteristicas(dummy_lms, 'Right')
        self.assertIn(len(feats), [63, 64])
        det.hands.close()

    def test_recolector_integration(self):
        """Verifica que recolector.py extraiga features y contadores desde gestures_common."""
        import recolector
        rec = recolector.RecolectorGestos()
        self.assertEqual(len(rec.contadores), 18)
        dummy_lms = [(float(i), float(i), 0.0) for i in range(21)]
        feats = rec.extraer_caracteristicas(dummy_lms)
        self.assertEqual(len(feats), 63)
        rec.hands.close()


if __name__ == '__main__':
    unittest.main()

