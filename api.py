"""
API REST - HAND2EMOJI
FastAPI + Uvicorn
Endpoint principal: POST /predict
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import os
import time

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODELS_DIR       = 'models'
CONFIANZA_MINIMA = 0.6

# ============================================================
# APP FASTAPI
# ============================================================
app = FastAPI(
    title="Hand2Emoji API",
    description="Detecta gestos de mano y retorna el emoji correspondiente",
    version="1.0.0"
)

# CORS — permite que la app web y OBS consuman la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODELOS (se cargan una sola vez al iniciar el servidor)
# ============================================================
class ModeloHandler:
    def __init__(self):
        print("🤖 Cargando modelos...")
        try:
            with open(os.path.join(MODELS_DIR, 'modelo.pkl'), 'rb') as f:
                self.modelo = pickle.load(f)
            with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(MODELS_DIR, 'labels.pkl'), 'rb') as f:
                self.le = pickle.load(f)
            with open(os.path.join(MODELS_DIR, 'metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)

            self.emoji_map = self.metadata['emoji_map']
            self.gestos    = self.metadata['gestos']
            print(f"✅ Modelos cargados | {len(self.gestos)} gestos")

        except FileNotFoundError as e:
            print(f"❌ Error cargando modelos: {e}")
            raise

    def predecir(self, landmarks: list, lado: str) -> dict:
        """
        Recibe 63 landmarks normalizados + lado
        Retorna gesto, emoji y confianza
        """
        lado_num = 1 if lado == 'Right' else 0
        features = landmarks + [lado_num]  # 64 features

        features_scaled = self.scaler.transform([features])

        # Probabilidades por clase
        proba  = self.modelo.predict_proba(features_scaled)[0]
        idx    = np.argmax(proba)
        confianza = float(proba[idx])
        gesto  = self.le.classes_[idx]

        # Top 3 predicciones (útil para debug)
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            {
                "gesto": self.le.classes_[i],
                "emoji": self.emoji_map.get(self.le.classes_[i], '🫱'),
                "confianza": round(float(proba[i]), 4)
            }
            for i in top3_idx
        ]

        return {
            "gesto":     gesto,
            "emoji":     self.emoji_map.get(gesto, '🫱'),
            "confianza": round(confianza, 4),
            "detectado": confianza >= CONFIANZA_MINIMA,
            "top3":      top3,
        }


# Instancia global del handler
handler = ModeloHandler()


# ============================================================
# SCHEMAS (estructura de los datos de entrada y salida)
# ============================================================
class LandmarksInput(BaseModel):
    """
    Entrada del endpoint /predict

    - landmarks: lista de 63 floats (x0..x20, y0..y20, z0..z20)
                 ya normalizados respecto a la muñeca
    - lado: 'Left' o 'Right'
    """
    landmarks: list[float]
    lado: str

    class Config:
        json_schema_extra = {
            "example": {
                "landmarks": [0.0] * 63,
                "lado": "Right"
            }
        }


class PrediccionOutput(BaseModel):
    gesto:     str
    emoji:     str
    confianza: float
    detectado: bool
    top3:      list
    tiempo_ms: float


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Endpoint de bienvenida — confirma que la API está viva"""
    return {
        "mensaje": "Hand2Emoji API funcionando! 🤌",
        "version": "1.0.0",
        "gestos_disponibles": len(handler.gestos),
        "docs": "/docs"
    }


@app.get("/gestos")
def listar_gestos():
    """Lista todos los gestos que el modelo puede detectar"""
    return {
        "total": len(handler.gestos),
        "gestos": [
            {
                "nombre": g,
                "emoji": handler.emoji_map.get(g, '🫱')
            }
            for g in sorted(handler.gestos)
        ]
    }


@app.post("/predict", response_model=PrediccionOutput)
def predecir(data: LandmarksInput):
    """
    Endpoint principal — recibe landmarks y retorna el gesto detectado.

    Entrada:
    - landmarks: 63 floats normalizados (x0-x20, y0-y20, z0-z20)
    - lado: 'Left' o 'Right'

    Salida:
    - gesto: nombre del gesto detectado
    - emoji: emoji correspondiente
    - confianza: probabilidad entre 0 y 1
    - detectado: true si confianza >= 0.6
    - top3: las 3 mejores predicciones
    - tiempo_ms: tiempo de inferencia en milisegundos
    """
    # Validaciones
    if len(data.landmarks) != 63:
        raise HTTPException(
            status_code=422,
            detail=f"Se esperan 63 landmarks, se recibieron {len(data.landmarks)}"
        )
    if data.lado not in ('Left', 'Right'):
        raise HTTPException(
            status_code=422,
            detail="El campo 'lado' debe ser 'Left' o 'Right'"
        )

    # Predicción
    t0 = time.perf_counter()
    resultado = handler.predecir(data.landmarks, data.lado)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {**resultado, "tiempo_ms": round(elapsed_ms, 2)}


@app.get("/health")
def health():
    """Health check — para monitoreo"""
    return {
        "status": "ok",
        "modelo_cargado": handler.modelo is not None,
        "n_gestos": len(handler.gestos)
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("🚀 Iniciando Hand2Emoji API")
    print("=" * 55)
    print("  Docs:   http://localhost:8000/docs")
    print("  Health: http://localhost:8000/health")
    print("  Gestos: http://localhost:8000/gestos")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)