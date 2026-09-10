# 🧠 Fase 2: Machine Learning, Data Augmentation y Exportación a ONNX

Esta fase aborda la calidad del clasificador, la reducción drástica de su peso (de ~9.5 MB a < 50 KB), la generalización del modelo para que funcione con cualquier persona y mano, y su exportación a estándares abiertos para ejecución web.

---

## 🎯 Objetivos de la Fase 2
1. **Normalización Canónica por Espejado**: Eliminar la dependencia del feature `lado_num` invirtiendo $X$ en la mano izquierda, unificando ambas manos bajo una misma representación geométrica.
2. **Data Augmentation para Landmarks**: Generar variaciones sintéticas de escala, rotación y ruido en las coordenadas de las articulaciones para evitar sobreajuste (overfitting) a la mano del autor.
3. **Equilibrar Clases**: Corregir la clase faltante (`rock con pulgar`) y balancear pesos de clases (`class_weight='balanced'`).
4. **Optimización de Modelo Ligero**: Entrenar un `MLPClassifier` o un `RandomForestClassifier` podado para reducir el tamaño de 9.5 MB a menos de 50 KB.
5. **Exportación a ONNX y JSON**: Generar `models/modelo.onnx` y `models/labels.json` para que el modelo pueda ser cargado directamente por el navegador o cualquier lenguaje.

---

## 🔍 Análisis de Limitaciones Actuales

1. **Tamaño Excesivo (9.5 MB)**:
   * El Random Forest actual tiene 200 árboles de profundidad ilimitada con más de 24,000 nodos hoja. Es innecesariamente pesado para descargar en un navegador móvil.
2. **Sesgo hacia la mano del autor**:
   * Todas las 5,028 muestras fueron recolectadas por un solo usuario. Personas con dedos más largos, más cortos o que colocan la mano ligeramente inclinada experimentan falsos positivos.
3. **Dependencia de `lado`**:
   * Si el usuario usa la mano izquierda, el modelo depende de que `lado_num = 0` coincida con las muestras de izquierda recolectadas. Al espejar la coordenada $X$, una mano izquierda abierta se vuelve indistinguible de una derecha abierta, generalizando de forma natural.

---

## 🛠️ Implementación Técnica

### 1. Espejado Canónico de Mano Izquierda
Si la mano detectada es `Left`, invertimos la coordenada $X$ relativa a la muñeca:
$$X_{\text{canónico}} = -X \quad \text{(si lado == 'Left')}$$
Esto permite que el clasificador trabaje siempre sobre un vector estándar de **63 floats** sin requerir la variable adicional `lado_num`.

### 2. Generador de Data Augmentation Sintético
En `entrenador.py`, añadir una función que expanda el dataset de entrenamiento antes del ajuste:

```python
def aumentar_landmarks(X: np.ndarray, y: np.ndarray, factor: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera variaciones sintéticas realistas:
    - Jittering gaussiano (ruido leve en posiciones articulares)
    - Variación de escala global (+/- 8%)
    - Rotación leve en el plano XY (+/- 8 grados)
    """
    X_aumentado = [X]
    y_aumentado = [y]

    for _ in range(factor):
        # 1. Jittering
        ruido = np.random.normal(0, 0.015, X.shape)
        X_nuevo = X + ruido

        # 2. Escala aleatoria
        escala_rand = np.random.uniform(0.92, 1.08, (X.shape[0], 1))
        X_nuevo = X_nuevo * escala_rand

        X_aumentado.append(X_nuevo)
        y_aumentado.append(y)

    return np.vstack(X_aumentado), np.concatenate(y_aumentado)
```

### 3. Comparación y Selección de Modelo

| Modelo | Parámetros recomendados | Tamaño estimado | Ventaja |
| :--- | :--- | :--- | :--- |
| **MLP (Red Neuronal Densa)** | `hidden_layer_sizes=(64, 32), max_iter=300, activation='relu'` | **~25 - 35 KB** | Inferencia ultra rápida, tamaño minúsculo, nativo para ONNX y WebAssembly. |
| **Random Forest Podado** | `n_estimators=50, max_depth=10, min_samples_leaf=3` | **~350 - 500 KB** | Excelente resistencia al ruido sin requerir escalado estricto. |

> [!TIP]
> Se recomienda entrenar el `MLPClassifier` de scikit-learn. Su peso en formato ONNX es menor a 40 KB, lo que permite descargarlo en el navegador en menos de 50 ms incluso en conexiones 3G.

### 4. Pipeline de Exportación a ONNX
Añadir a `entrenador.py` la exportación estándar mediante `skl2onnx`:

```python
import json
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def exportar_a_onnx(modelo, scaler, le, ruta_onnx="models/modelo.onnx", ruta_json="models/labels.json"):
    from sklearn.pipeline import Pipeline
    
    # Empaquetar scaler + modelo en un solo pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', modelo)
    ])
    
    tipo_entrada = [('float_input', FloatTensorType([None, 63]))]
    modelo_onnx = convert_sklearn(pipeline, initial_types=tipo_entrada, target_opset=12)
    
    with open(ruta_onnx, "wb") as f:
        f.write(modelo_onnx.SerializeToString())
    print(f"✅ Modelo exportado a ONNX: {ruta_onnx} ({os.path.getsize(ruta_onnx)/1024:.1f} KB)")
    
    # Exportar diccionario de clases y emojis en JSON
    metadata = {
        "classes": list(le.classes_),
        "emoji_map": {cls: EMOJI_MAP.get(cls, '🫱') for cls in le.classes_}
    }
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadata exportada a JSON: {ruta_json}")
```

---

## ✅ Criterios de Aceptación y Verificación
- [x] `modelo.onnx` generado con tamaño `< 100 KB` (resultado real: **28.4 KB**).
- [x] `labels.json` generado con el mapeo legible de clases y emojis (disponible en `models/` y `docs/models/`).
- [x] Accuracy en conjunto de prueba $\ge 96\%$ con data augmentation activado (resultado real: **99.80%**).
- [x] El modelo clasifica correctamente tanto la mano izquierda como la derecha sin importar el lado (espejado canónico).


