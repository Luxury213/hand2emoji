"""
test_phase3.py - Suite de pruebas automatizadas para la Fase 3
Verifica la integración de ONNX Runtime Web en docs/index.html,
la disponibilidad y validez de los activos estáticos para GitHub Pages,
y el contrato de tensores numéricos para WebAssembly.
"""

import os
import sys
import json
import unittest
import http.server
import socketserver
import threading
import urllib.request
import time
import onnx
import onnxruntime as ort
import numpy as np

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


class TestPhase3(unittest.TestCase):

    def test_index_html_onnx_integration(self):
        """Verifica que docs/index.html incluya las etiquetas y lógica de ONNX Runtime Web."""
        html_path = os.path.join('docs', 'index.html')
        self.assertTrue(os.path.exists(html_path), "docs/index.html debe existir")

        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()

        # Biblioteca ONNX Runtime Web CDN
        self.assertIn('onnxruntime-web', html, "Debe incluir ort.min.js de onnxruntime-web")

        # Funciones clave
        self.assertIn('initLocalEngine', html, "Debe incluir initLocalEngine()")
        self.assertIn('normalizeCanonico', html, "Debe incluir normalización canónica con espejado")
        self.assertIn('onnxSession', html, "Debe manejar la sesión ONNX")
        self.assertIn('renderChips', html, "Debe incluir función para renderizar chips de gestos")
        self.assertIn('DEFAULT_GESTOS', html, "Debe incluir lista inicial de gestos para visualización instantánea")

        # Throttle reducido para 60 FPS
        import re
        self.assertTrue(re.search(r'THROTTLE\s*=\s*40', html), "El throttle debe ser ~40ms para inferencia local fluida")

    def test_web_static_models_assets(self):
        """Verifica que modelo.onnx y labels.json estén disponibles en docs/models/."""
        model_path = os.path.join('docs', 'models', 'modelo.onnx')
        labels_path = os.path.join('docs', 'models', 'labels.json')

        self.assertTrue(os.path.exists(model_path), "docs/models/modelo.onnx debe existir")
        self.assertTrue(os.path.exists(labels_path), "docs/models/labels.json debe existir")

        # Tamaño del modelo ONNX web
        size_kb = os.path.getsize(model_path) / 1024
        print(f"\n   📦 Tamaño verificado docs/models/modelo.onnx: {size_kb:.1f} KB")
        self.assertLess(size_kb, 50.0, "El modelo debe pesar menos de 50 KB para carga web instantánea")

        # Validez del modelo ONNX
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        # Validez del archivo labels.json
        with open(labels_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        self.assertEqual(meta['n_features'], 63)
        self.assertEqual(len(meta['classes']), 17)
        self.assertEqual(len(meta['emoji_map']), 17)

    def test_onnx_tensor_contract_for_web(self):
        """Verifica que el modelo ONNX exporte tensores puros sin zipmap para consumo directo en JS."""
        model_path = os.path.join('docs', 'models', 'modelo.onnx')
        sess = ort.InferenceSession(model_path)

        output_names = [out.name for out in sess.get_outputs()]
        self.assertIn('label', output_names, "Debe tener salida 'label'")
        self.assertIn('probabilities', output_names, "Debe tener salida 'probabilities'")

        # Inferencia con dummy input de 63 floats
        dummy_input = np.zeros((1, 63), dtype=np.float32)
        outputs = sess.run(None, {'float_input': dummy_input})

        labels = outputs[0]
        probs = outputs[1]

        self.assertEqual(labels.shape, (1,))
        self.assertEqual(probs.shape, (1, 17))
        self.assertIsInstance(probs, np.ndarray, "Las probabilidades deben ser un ndarray puro (sin zipmap)")

        # Comprobar que sumen ~1.0 (softmax)
        self.assertAlmostEqual(float(np.sum(probs[0])), 1.0, places=3)

    def test_http_serving_simulation(self):
        """Simula el servidor web estático de GitHub Pages y verifica peticiones HTTP 200."""
        port = 8991
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(('127.0.0.1', port), handler)

        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        time.sleep(0.3)

        try:
            # 1. HTML principal
            res = urllib.request.urlopen(f'http://127.0.0.1:{port}/docs/index.html')
            self.assertEqual(res.status, 200)

            # 2. labels.json
            res = urllib.request.urlopen(f'http://127.0.0.1:{port}/docs/models/labels.json')
            self.assertEqual(res.status, 200)
            data = json.loads(res.read().decode('utf-8'))
            self.assertEqual(len(data['classes']), 17)

            # 3. modelo.onnx
            res = urllib.request.urlopen(f'http://127.0.0.1:{port}/docs/models/modelo.onnx')
            self.assertEqual(res.status, 200)
            model_bytes = res.read()
            self.assertGreater(len(model_bytes), 10000)

        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == '__main__':
    unittest.main()
