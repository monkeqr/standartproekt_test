from pydantic import BaseModel
from typing import List, Union

class RegnoInput(BaseModel):
    regno_recognize: str
    afts_regno_ai: str
    recognition_accuracy: float
    afts_regno_ai_score: float
    # В CSV это строки вида "[0.99, ...]", но в JSON лучше принимать списки.
    # Сервис поддержит и то, и то благодаря логике внутри logic.py
    afts_regno_ai_char_scores: str 
    afts_regno_ai_length_scores: str
    camera_type: str
    camera_class: str
    time_check: str
    direction: int

class PredictionResponse(BaseModel):
    regno_recognize: str
    prediction_proba: List[List[float]] # CatBoost возвращает вероятности классов