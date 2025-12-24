import numpy as np
import re
import pandas as pd
import calendar
from catboost import CatBoostClassifier

# Константы вынесены в глобальную область видимости модуля
AI_SYM = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'D', 'А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х']
OLD_CYRILLIC_SYM = ['И', 'Й', 'Ц', 'Г', 'Ш', 'Щ', 'З', 'Ъ', 'Ф', 'Ы', 'П', 'Л', 'Д', 'Ж', 'Э', 'Я', 'Ч', 'Ь', 'Б', 'Ю']
LAT_SYM = ['Q', 'W', 'R', 'Y', 'U', 'I', 'S', 'F', 'G', 'J', 'L', 'Z', 'V', 'N']

class RegnoPredictor:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}...")
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        print("Model loaded successfully.")

    def _d(self, alist, blist):
        zipped = (zip(alist, blist))
        um1 = dict(sorted(zipped, key=lambda t: t[1], reverse=True))
        return um1

    def _get_ai_syms(self, nn_regno, nn_sym_scores):
        letters = self._d(nn_regno, nn_sym_scores)
        for s in AI_SYM:
            if s not in letters:
                letters[s] = 1
        return letters

    def _regno_category(self, regno):
        if re.fullmatch('^[А-Я]\d{3}[А-Я]{2}\d{2}(|\d)$', regno): return 'private car'
        elif re.fullmatch('^[А-Я]{2}\d{5}(|\d)$', regno): return ' lk_taxi_r_pricep'
        elif re.fullmatch('^[А-Я]{2}\d{6}(|\d)$', regno): return 'lk_pricep_r_transit'
        elif re.fullmatch('^\d{4}[А-Я]{2}\d{2}(|\d)$', regno): return 'lk_moto_tract'
        elif re.fullmatch('^[А-Я]{1}\d{4}[А-Я]{2}$', regno): return 'some'
        elif re.fullmatch('^[А-Я]{2}\d{3}[А-Я]\d{2}(|\d)$', regno): return 'lk_transit'
        elif re.fullmatch('^(T|Т)[А-Я]{2}\d{5}(|\d)$', regno): return 'lk_export'
        elif re.fullmatch('^[А-Я]\d{6}(|\d)$', regno): return 'lk_mvd_avto'
        elif re.fullmatch('^\d{4}[А-Я]\d{2}(|\d)$', regno): return 'lk_mvd_moto'
        elif re.fullmatch('^\d{3}[А-Я]\d{2}(|\d)$', regno): return 'lk_mvd_pricep'
        elif re.fullmatch('^\d{3}(CD|СD|D|T|Т)\d{3}(|\d)(|\d)(|\d)$', regno): return 'lk_diplomat'
        elif re.fullmatch('^\d{4}[А-Я]{2}\d{2}(|\d)$', regno): return 'lk_army'
        else: return 'unknown'

    def _count_foreign_syms(self, regno):
        i = 0
        for s in regno:
            if s not in (AI_SYM + OLD_CYRILLIC_SYM + LAT_SYM):
                i += 1
        return i # Функция в оригинале не возвращала значение, исправлено

    def _str_to_list(self, s):
        # Обработка строкового представления списка из CSV
        if isinstance(s, str):
            if s == '[]': return [0]
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            if not s: return [0]
            try:
                return [float(i) for i in s.split(',')]
            except ValueError:
                return [0]
        return s if isinstance(s, list) else [0]

    def predict(self, item):
        # item - объект Pydantic или словарь
        
        # Подготовка фичей
        nn_sym_scores = self._str_to_list(item.afts_regno_ai_char_scores)
        nn_len_scores = self._str_to_list(item.afts_regno_ai_length_scores)
        
        # Основная логика из pick_regno.py
        x = {}
        x['afts_regno_ai_score'] = item.afts_regno_ai_score
        x['direction'] = item.direction
        x['recognition_accuracy'] = item.recognition_accuracy
        
        # Агрегация score
        if len(nn_sym_scores) > 0:
            x['max_sym_score'] = np.max(nn_sym_scores)
            x['min_sym_score'] = np.min(nn_sym_scores)
        else:
             x['max_sym_score'] = 0
             x['min_sym_score'] = 0
             
        x['max_len_score'] = np.max(nn_len_scores) if len(nn_len_scores) > 0 else 0
        
        x['ai_len'] = len(item.afts_regno_ai)
        x['cam_len'] = len(item.regno_recognize)
        
        # В catboost признаки часто ожидаются строками, если это категориальные, 
        # но в скрипте использовался ' '.join. Оставим как есть.
        x['regno_recognize_text'] = ' '.join(list(item.regno_recognize))
        x['afts_regno_ai_text'] = ' '.join(list(item.afts_regno_ai))
        
        x['camera_type'] = item.camera_type
        x['camera_class'] = item.camera_class
        
        time_check = pd.to_datetime(item.time_check)
        x['weekday'] = calendar.day_name[time_check.dayofweek]
        x['month'] = calendar.month_name[time_check.month]
        x['hour'] = time_check.hour
        
        x['regno_template'] = self._regno_category(item.regno_recognize)
        x['regno_template_ai'] = self._regno_category(item.afts_regno_ai)
        x['foreign_sym'] = self._count_foreign_syms(item.regno_recognize)
        
        letters = self._get_ai_syms(item.afts_regno_ai, nn_sym_scores)
        x.update(letters)
        
        # Подготовка Series для модели
        features_data = pd.Series(x)
        
        # Важно: выбираем только те признаки, которые знает модель.
        # Обычно model.feature_names_ доступен после загрузки.
        # Здесь мы предполагаем, что словарь x содержит всё необходимое.
        
        y = self.model.predict_proba(features_data[self.model.feature_names_])
        return y