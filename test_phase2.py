"""
test_phase2.py - Suite de pruebas automatizadas para la Fase 2
Verifica normalización canónica, data augmentation, modelo ONNX, metadatos JSON
e inferencia en api.py y detector.py.
"""

import os
import sys
import json
import unittest
import numpy as np

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


class TestPhase2(unittest.TestCase):

    def test_canonical_mirroring(self):
        """Verifica que el espejado canónico invierta únicamente las coordenadas X para mano izquierda."""
        import gestures_common as gc

        dummy_landmarks = [(float(i + 1), float(i * 2 + 1), float(i * 0.1)) for i in range(21)]
        feat_der = gc.extraer_caracteristicas_canonicas(dummy_landmarks, 'Right')
        feat_izq = gc.extraer_caracteristicas_canonicas(dummy_landmarks, 'Left')

        self.assertEqual(len(feat_der), 63)
        self.assertEqual(len(feat_izq), 63)

        # Primeras 21 son X: deben ser inversas
        for i in range(21):
            self.assertAlmostEqual(feat_der[i], -feat_izq[i], places=5)

        # Siguientes 42 son Y y Z: deben ser idénticas
        for i in range(21, 63):
            self.assertAlmostEqual(feat_der[i], feat_izq[i], places=5)

    def test_data_augmentation(self):
        """Verifica la generación de datos sintéticos con rotación, escala y ruido."""
        import entrenador as ent

        N = 20
        X = np.random.uniform(-1.0, 1.0, size=(N, 63)).astype(np.float32)
        y = np.random.randint(0, 5, size=N)

        factor = 2
        X_aug, y_aug = ent.aumentar_datos(X, y, factor=factor, random_state=42)

        self.assertEqual(X_aug.shape[0], N * (factor + 1))
        self.assertEqual(X_aug.shape[1], 63)
        self.assertEqual(len(y_aug), N * (factor + 1))
        self.assertFalse(np.isnan(X_aug).any(), "No debe haber valores NaN")
        self.assertFalse(np.isinf(X_aug).any(), "No debe haber valores Inf")

    def test_onnx_model_exists_and_size(self):
        """Verifica que el modelo ONNX exista y su tamaño sea < 50 KB (reducción de 330x)."""
        import onnx

        ruta_onnx = os.path.join('models', 'modelo.onnx')
        self.assertTrue(os.path.exists(ruta_onnx), f"{ruta_onnx} debe existir")

        size_kb = os.path.getsize(ruta_onnx) / 1024
        print(f"\n   📦 Tamaño verificado modelo.onnx: {size_kb:.1f} KB")
        self.assertLess(size_kb, 50.0, "El modelo ONNX debe pesar menos de 50 KB")

        # Cargar con biblioteca oficial ONNX
        model = onnx.load(ruta_onnx)
        onnx.checker.check_model(model)

        # Verificar copia en docs/models para GitHub Pages
        ruta_docs_onnx = os.path.join('docs', 'models', 'modelo.onnx')
        self.assertTrue(os.path.exists(ruta_docs_onnx), "docs/models/modelo.onnx debe existir para web")

    def test_labels_json(self):
        """Verifica que labels.json contenga metadatos válidos para la web."""
        for dir_path in ['models', os.path.join('docs', 'models')]:
            ruta_json = os.path.join(dir_path, 'labels.json')
            self.assertTrue(os.path.exists(ruta_json), f"{ruta_json} debe existir")

            with open(ruta_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.assertEqual(data['n_features'], 63)
            self.assertEqual(len(data['classes']), 17)
            self.assertIn('corazon', data['classes'])
            self.assertIn('rock', data['classes'])
            self.assertEqual(len(data['emoji_map']), 17)
            self.assertTrue(data.get('canonical_mirroring', False))

    def test_api_with_phase2_model(self):
        """Verifica que api.py realice inferencias tanto con mano izquierda como derecha."""
        import api
        from api import LandmarksInput, predecir

        # Entrada mano derecha
        req_der = LandmarksInput(landmarks=[0.0] * 63, lado='Right')
        res_der = predecir(req_der)
        self.assertIn('gesto', res_der)
        self.assertIn('emoji', res_der)
        self.assertEqual(len(res_der['top3']), 3)

        # Entrada mano izquierda
        req_izq = LandmarksInput(landmarks=[0.0] * 63, lado='Left')
        res_izq = predecir(req_izq)
        self.assertIn('gesto', res_izq)
        self.assertIn('emoji', res_izq)
        self.assertEqual(len(res_izq['top3']), 3)

    def test_detector_with_phase2_model(self):
        """Verifica que detector.py inicialice y extraiga 63 features canónicas."""
        import detector
        det = detector.DetectorGestos()

        dummy_lms = [(float(i), float(i), 0.0) for i in range(21)]
        feats_der = det.extraer_caracteristicas(dummy_lms, 'Right')
        feats_izq = det.extraer_caracteristicas(dummy_lms, 'Left')

        self.assertEqual(len(feats_der), 63)
        self.assertEqual(len(feats_izq), 63)

        gesto, conf = det.predecir(feats_der)
        self.assertIsInstance(conf, float)
        det.hands.close()


if __name__ == '__main__':
    unittest.main()

