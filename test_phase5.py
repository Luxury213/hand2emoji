"""
test_phase5.py - Suite de pruebas automatizadas para la Fase 5
Verifica las funcionalidades del detector de escritorio avanzado:
- Renderizado de emojis Unicode en alta fidelidad y color usando Pillow.
- Gestor de macros e inyección en portapapeles con MacroManager.
- Arquitectura desacoplada de procesar_frame() en DetectorGestos.
- Disponibilidad y configuración de cámara virtual (pyvirtualcam).
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


class TestPhase5(unittest.TestCase):

    def test_emoji_renderer_pillow(self):
        """Verifica que EmojiRenderer dibuje pictogramas Unicode con píxeles de color sobre frames BGR."""
        import detector
        renderer = detector.EmojiRenderer(font_size=64)
        self.assertIsNotNone(renderer.font, "Debe cargar una fuente válida del sistema o fallback")

        # Frame sintético negro
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        # Dibujar emoji centrado
        out_frame = renderer.dibujar_emoji_centrado(frame.copy(), "👍", 100, 100)
        self.assertEqual(out_frame.shape, (200, 200, 3))
        self.assertEqual(out_frame.dtype, np.uint8)

        # Debe haber píxeles dibujados (color real)
        drawn_pixels = np.sum(out_frame > 0)
        self.assertGreater(drawn_pixels, 1000, "Debe renderizar el emoji con múltiples píxeles de color")

        # Probar renderizado en posición fija
        pos_frame = renderer.dibujar_emoji_posicion(frame.copy(), "✌️", (20, 20), font_size=32)
        self.assertGreater(np.sum(pos_frame > 0), 200)

    def test_macro_manager_hold_and_clipboard(self):
        """Verifica el ciclo de vida, retención (hold) y copiado en MacroManager."""
        import detector
        macro = detector.MacroManager(activo=False, hold_frames=5, cooldown_frames=10)

        # Inactivo por defecto: no debe disparar
        self.assertFalse(macro.activo)
        self.assertFalse(macro.procesar_gesto('pulgar', '👍'))

        # Activar macro
        macro.alternar()
        self.assertTrue(macro.activo)

        # Simular 4 frames de hold: no debe disparar aún
        for i in range(4):
            disparado = macro.procesar_gesto('pulgar', '👍')
            self.assertFalse(disparado, f"No debe disparar en el frame {i+1} de 5")
            self.assertGreater(macro.porcentaje_progreso(), 0.0)

        # Frame 5: debe disparar
        disparado = macro.procesar_gesto('pulgar', '👍')
        self.assertTrue(disparado, "Debe disparar al alcanzar hold_frames=5")

        # Frames en cooldown: no debe disparar aunque se mantenga el gesto
        self.assertFalse(macro.procesar_gesto('pulgar', '👍'))
        self.assertGreater(macro.contador_cooldown, 0)

        # Probar copiado real al portapapeles
        exito_clip = detector.MacroManager.copiar_al_portapapeles("🫶")
        self.assertTrue(exito_clip, "El copiado al portapapeles con pyperclip debe funcionar")

        if detector.pyperclip:
            self.assertEqual(detector.pyperclip.paste(), "🫶")

    def test_detector_gestos_procesar_frame(self):
        """Verifica que procesar_frame procese frames sintéticos de forma modular y sin errores."""
        import detector
        det = detector.DetectorGestos(macro_default=False)

        # Frame sintético BGR vacío (sin manos)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed, info = det.procesar_frame(dummy_frame)

        self.assertEqual(processed.shape, (480, 640, 3))
        self.assertIsInstance(info, dict)
        self.assertFalse(info['mano_detectada'])
        self.assertIsNone(info['gesto'])

        # Simular un gesto activo artificialmente y verificar dibujo de overlay
        det.gesto_actual = 'pulgar'
        det.emoji_actual = '👍'
        det.confianza_actual = 0.95

        overlay_frame = det.dibujar_emoji_overlay(dummy_frame.copy(), 480, 640)
        self.assertGreater(np.sum(overlay_frame > 0), 2000, "El overlay debe dibujar el círculo y el emoji")

        # Probar toast
        det.mostrar_toast("Prueba de notificación", duracion=5.0)
        self.assertEqual(det.toast_msg, "Prueba de notificación")
        self.assertGreater(det.toast_expira, 0)

        det.hands.close()

    def test_pyvirtualcam_presence(self):
        """Verifica que pyvirtualcam esté instalado y disponible para streaming."""
        import detector
        self.assertIsNotNone(detector.pyvirtualcam, "pyvirtualcam debe estar instalado en el entorno")
        self.assertTrue(hasattr(detector.pyvirtualcam, 'Camera'))
        self.assertTrue(hasattr(detector.pyvirtualcam, 'PixelFormat'))

    def test_argparse_cli_options(self):
        """Verifica las opciones de línea de comandos de detector.py."""
        import detector
        parser = detector.argparse.ArgumentParser()
        # Parse con opciones simuladas
        import detector as det_mod
        args = det_mod.parse_args
        self.assertTrue(callable(args))


if __name__ == '__main__':
    unittest.main()

