"""
test_phase4.py - Suite de pruebas automatizadas para la Fase 4
Verifica la implementación de la experiencia frontend minimalista en docs/index.html:
- Ausencia de artefactos de diseño vibecoded (neón falso, esquinas recargadas).
- Presencia del Emoji Composer y gestión del buffer.
- Síntesis de retroalimentación sonora con Web Audio API.
- Selectores dinámicos de sensibilidad (HOLD_MS).
- Modos de auto-copiado, toast efímero y soporte de pantalla completa.
- Servido HTTP e integridad de la integración.
"""

import os
import sys
import json
import re
import unittest
import http.server
import socketserver
import threading
import urllib.request
import time

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


class TestPhase4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.html_path = os.path.join('docs', 'index.html')
        with open(cls.html_path, 'r', encoding='utf-8') as f:
            cls.html = f.read()

    def test_minimalist_design_tokens(self):
        """Verifica tipografía, paleta monocromática y ausencia de clichés vibecoded."""
        html = self.html

        # Tipografías sobrias y profesionales
        self.assertIn('Plus Jakarta Sans', html, "Debe utilizar Plus Jakarta Sans como tipografía principal")
        self.assertIn('JetBrains Mono', html, "Debe utilizar JetBrains Mono para telemetría y chips técnicos")

        # Ausencia de artefactos vibecoded / cyberpunk falso
        self.assertNotIn('.corner', html, "No debe contener esquinas cyberpunk decorativas (.corner)")
        self.assertNotIn('.corner-tl', html, "No debe contener .corner-tl")
        self.assertNotIn('.corner-tr', html, "No debe contener .corner-tr")
        self.assertNotIn('.corner-bl', html, "No debe contener .corner-bl")
        self.assertNotIn('.corner-br', html, "No debe contener .corner-br")

        # Paleta monocromática de alto contraste y superficie frosted glass
        self.assertIn('--bg: #050507', html, "Debe utilizar paleta carbon deep (#050507)")
        self.assertIn('backdrop-filter', html, "Debe usar glassmorphism auténtico con backdrop-filter")

    def test_emoji_composer_elements_and_logic(self):
        """Verifica los componentes DOM y la lógica del teclado/compositor de emojis."""
        html = self.html

        # Elementos del DOM
        self.assertIn('id="composerOutput"', html, "Debe contener el contenedor composerOutput")
        self.assertIn('id="btnCopyBuffer"', html, "Debe contener el botón de copiar buffer")
        self.assertIn('id="btnBackBuffer"', html, "Debe contener el botón de borrar último emoji (backspace)")
        self.assertIn('id="btnClearBuffer"', html, "Debe contener el botón de vaciar buffer")

        # Lógica en JavaScript
        self.assertIn('let emojiBuffer = [];', html, "Debe declarar el array emojiBuffer")
        self.assertIn('function renderComposer()', html, "Debe contener la función renderComposer()")
        self.assertIn('function appendToComposer(', html, "Debe contener appendToComposer()")
        self.assertIn('navigator.clipboard.writeText', html, "Debe permitir el copiado nativo al portapapeles")

    def test_web_audio_api_haptic_feedback(self):
        """Verifica la síntesis de audio on-device sin archivos mp3 externos."""
        html = self.html

        # Web Audio API
        self.assertIn('AudioContext', html, "Debe inicializar o referenciar AudioContext")
        self.assertIn('createOscillator', html, "Debe sintetizar audio mediante oscilador")
        self.assertIn('createGain', html, "Debe controlar la ganancia exponencialmente")
        self.assertIn('playAcousticPop', html, "Debe definir playAcousticPop() para confirmación háptica")
        self.assertIn('playSubtleClick', html, "Debe definir playSubtleClick() para interacción con botones")

        # Botón de alternar audio
        self.assertIn('id="btnAudio"', html, "Debe contener el botón para activar/desactivar audio")
        self.assertIn('let soundActive = true;', html, "El audio debe estar activo por defecto")

    def test_dynamic_sensitivity_and_preferences(self):
        """Verifica los controles de velocidad (HOLD_MS), auto-copiado y pantalla completa."""
        html = self.html

        # Botones segmentados de sensibilidad
        self.assertIn('class="seg-btn', html, "Debe contener botones segmentados (.seg-btn)")
        self.assertIn('data-ms="550"', html, "Debe incluir opción rápida (550ms)")
        self.assertIn('data-ms="850"', html, "Debe incluir opción normal (850ms)")
        self.assertIn('data-ms="1300"', html, "Debe incluir opción segura (1300ms)")

        # Auto-copy toggle
        self.assertIn('id="btnAutoCopy"', html, "Debe tener botón de auto-copy")
        self.assertIn('let autoCopyActive = false;', html, "Auto-copy debe comenzar inactivo")

        # Pantalla completa
        self.assertIn('id="btnFullscreen"', html, "Debe tener botón de pantalla completa")
        self.assertIn('requestFullscreen', html, "Debe invocar requestFullscreen()")

        # Toast notification
        self.assertIn('id="toast"', html, "Debe existir elemento toast")
        self.assertIn('function showToast(', html, "Debe existir la función showToast()")

    def test_surgical_landmark_rendering(self):
        """Verifica el estilo visual refinado y discreto para los landmarks de MediaPipe."""
        html = self.html

        # Skeleton minimalista (conectores delgados y radio de nodo pequeño)
        self.assertIn('drawConnectors', html, "Debe utilizar drawConnectors")
        self.assertIn('drawLandmarks', html, "Debe utilizar drawLandmarks")
        self.assertIn('radius: 2.2', html, "Los puntos de landmarks deben tener un radio sutil (2.2px)")

    def test_http_serving_and_page_rendering(self):
        """Prueba de integración HTTP sirviendo docs/index.html y validando cabeceras y contenido."""
        port = 8992
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(('127.0.0.1', port), handler)

        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        time.sleep(0.3)

        try:
            url = f'http://127.0.0.1:{port}/docs/index.html'
            with urllib.request.urlopen(url) as response:
                self.assertEqual(response.status, 200)
                content_type = response.headers.get('Content-Type', '')
                self.assertIn('text/html', content_type)
                body = response.read().decode('utf-8')
                self.assertIn('Hand2Emoji', body)
                self.assertIn('composerOutput', body)
                self.assertIn('playAcousticPop', body)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == '__main__':
    unittest.main()
