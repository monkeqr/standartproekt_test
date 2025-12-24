from fastapi import FastAPI, HTTPException
from typing import List
import uvicorn
import os

from logic import RegnoPredictor
from schemas import RegnoInput, PredictionResponse

app = FastAPI(title="Regno ML Service", description="High-load API for regno recognition")

# Глобальная переменная для предиктора
predictor = None
MODEL_PATH = "micromodel.cbm"

@app.on_event("startup")
def load_model():
    global predictor
    if os.path.exists(MODEL_PATH):
        predictor = RegnoPredictor(MODEL_PATH)
    else:
        print(f"Warning: {MODEL_PATH} not found. Prediction will fail.")

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(items: List[RegnoInput]):
    """
    Принимает массив данных, возвращает массив предсказаний.
    Оптимизировано для высокой нагрузки.
    """
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    # В идеале CatBoost поддерживает batch predict (model.predict(DataFrame)).
    # Для максимальной скорости лучше собрать DataFrame из всех items и вызвать predict один раз.
    # Но для сохранения логики pick_regno (где много препроцессинга на одну строку)
    # сделаем итерацию, так как препроцессинг сложный.
    
    for item in items:
        try:
            proba = predictor.predict(item)
            results.append(PredictionResponse(
                regno_recognize=item.regno_recognize,
                prediction_proba=proba.tolist()
            ))
        except Exception as e:
            # Логируем ошибку, но не роняем весь батч (опционально)
            print(f"Error processing item {item.regno_recognize}: {e}")
            results.append(PredictionResponse(
                regno_recognize=item.regno_recognize,
                prediction_proba=[[]] 
            ))
            
    return results

@app.get("/health")
def health_check():
    if predictor:
        return {"status": "ok", "model": "loaded"}
    return {"status": "error", "model": "not loaded"}

if __name__ == "__main__":
    # Для локального запуска и дебага
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)