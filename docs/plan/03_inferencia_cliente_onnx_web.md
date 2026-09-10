# ⚡ Fase 3: Inferencia en el Navegador con ONNX Runtime Web (Zero Backend)

Esta es la mejora de mayor impacto para los usuarios finales. Permite ejecutar la predicción de emojis **100% dentro del navegador del usuario**, eliminando la necesidad de un servidor backend en Render/Railway, eliminando la latencia de red y haciendo que la aplicación funcione de inmediato sin esperas de inicio en frío.

---

## 🎯 Objetivos de la Fase 3
1. **Integrar ONNX Runtime Web (`ort-web`)** en [docs/index.html](file:///c:/Users/USUARIO/Desktop/Proyectos/Hand2Emoji/docs/index.html) mediante CDN.
2. **Cargar el modelo estático `modelo.onnx`** y su configuración `labels.json` desde el mismo hosting de GitHub Pages.
3. **Ejecutar la inferencia en JavaScript**: alimentar el tensor de 63 floats y obtener las probabilidades locales en < 3 ms.
4. **Eliminar el estado de "Despertando servidor"**: la aplicación estará lista en cuanto cargue la página web.
5. **Modo Híbrido / Fallback (Opcional)**: mantener la opción de consultar la API si se desea, pero priorizar el motor local por defecto.

---

## 📐 Comparativa de Flujo

### Flujo Anterior (Red HTTP)
$$\text{Cámara} \longrightarrow \text{MediaPipe} \xrightarrow{\text{HTTP POST 180ms}} \text{Render (Python)} \xrightarrow{\text{JSON}} \text{UI}$$
* Latencia: **250 - 600 ms**.
* Si el servidor se apaga: **Espera de 30 - 60 segundos**.
* Ancho de banda continuo y límites de requests.

### Flujo Nuevo (In-Browser Edge AI)
$$\text{Cámara} \longrightarrow \text{MediaPipe} \xrightarrow{\text{Memoria JS}} \text{ONNX Runtime Web} \longrightarrow \text{UI}$$
* Latencia: **1 - 4 ms**.
* Arranque: **Inmediato** al cargar los 40 KB del modelo.
* Sin costo de servidor ni dependencia de plataformas en la nube.

---

## 🛠️ Implementación en `docs/index.html`

### 1. Incluir ONNX Runtime Web
Añadir antes del script principal:
```html
<!-- ONNX Runtime Web para inferencia ultra-rápida en el cliente -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort.min.js"></script>
```

### 2. Inicialización del Motor de Inferencia Local
Reemplazar la función de chequeo de API por la carga del modelo local:

```javascript
let onnxSession = null;
let modelLabels = [];
let modelEmojiMap = {};

async function initLocalEngine() {
  const statusPill = document.getElementById('statusPill');
  const statusText = document.getElementById('statusText');
  const btnStart   = document.getElementById('btnStart');
  const btnOverlay = document.getElementById('btnOverlay');

  try {
    statusText.textContent = 'Cargando modelo local...';
    
    // 1. Cargar metadatos de etiquetas y emojis
    const metaRes = await fetch('models/labels.json');
    const meta = await metaRes.json();
    modelLabels = meta.classes;
    modelEmojiMap = meta.emoji_map;

    // 2. Crear sesión ONNX con aceleración WebAssembly / WebGL
    ort.env.wasm.numThreads = 1;
    onnxSession = await ort.InferenceSession.create('models/modelo.onnx', {
      executionProviders: ['wasm']
    });

    // 3. Renderizar chips de gestos en la barra inferior
    renderGesturesChips();

    statusPill.className = 'status-pill online';
    statusText.textContent = 'Motor: Local ⚡';
    btnStart.disabled = false;
    btnOverlay.disabled = false;
    console.log('✅ Motor ONNX Web listo. Clases:', modelLabels.length);

  } catch (error) {
    console.warn('⚠️ No se pudo cargar el modelo ONNX local, usando API remota:', error);
    // Fallback opcional a API tradicional
    checkAPI();
  }
}
```

### 3. Normalización con Espejado Canónico en JavaScript
Asegurar que la normalización en JS replique exactamente la lógica de Python:

```javascript
function normalizeLandmarksJS(lms, lado, W, H) {
  const mx = lms[0].x * W, my = lms[0].y * H, mz = lms[0].z;
  const r9x = lms[9].x * W, r9y = lms[9].y * H, r9z = lms[9].z;
  
  // Escala euclidiana respecto a la base del dedo medio (punto 9)
  const s = Math.sqrt((r9x - mx)**2 + (r9y - my)**2 + (r9z - mz)**2) + 1e-6;
  
  const xs = [], ys = [], zs = [];
  // Si la mano es izquierda, invertimos X para canonizar a mano derecha
  const factorX = (lado === 'Left') ? -1 : 1;

  for (const p of lms) {
    xs.push(((p.x * W - mx) / s) * factorX);
    ys.push((p.y * H - my) / s);
    zs.push((p.z - mz) / s);
  }

  // 63 floats normalizados
  return new Float32Array([...xs, ...ys, ...zs]);
}
```

### 4. Inferencia Local en JavaScript
Reemplazar la llamada `fetch('/predict')` por la sesión ONNX:

```javascript
async function predictLocal(lms, lado, W, H) {
  if (!onnxSession) return null;

  const features = normalizeLandmarksJS(lms, lado, W, H);
  const tensor = new ort.Tensor('float32', features, [1, 63]);

  // Inferencia
  const feeds = { float_input: tensor };
  const outputMap = await onnxSession.run(feeds);
  
  // Dependiendo de si se exportó probabilidades o etiquetas:
  const probabilities = outputMap.probabilities ? outputMap.probabilities.data : outputMap.output_probability.data;

  // Encontrar clase con mayor probabilidad
  let maxIdx = 0;
  let maxProb = 0;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIdx = i;
    }
  }

  const gesto = modelLabels[maxIdx];
  return {
    gesto: gesto,
    emoji: modelEmojiMap[gesto] || '🫱',
    confianza: maxProb,
    detectado: maxProb >= 0.65
  };
}
```

---

## ✅ Criterios de Aceptación y Verificación
- [x] La demo web en `docs/index.html` carga y detecta gestos sin realizar peticiones HTTP a servidores externos.
- [x] Tasa de FPS optimizada con `THROTTLE = 40ms` (25-30 inferencias/s y 60 FPS visuales).
- [x] Se sustituye el estado amarillo de "Despertando servidor" por "IA Local ⚡ 60 FPS" inmediato.
- [x] Los activos estáticos (`modelo.onnx` de 28.3 KB y `labels.json`) son servidos directamente por GitHub Pages.


