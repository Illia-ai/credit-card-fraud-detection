# Обнаружение мошенничества с кредитными картами

## Описание задачи
Классификация транзакций как нормальных или мошеннических на основе датасета Credit Card Fraud Detection (Kaggle).  
**Дисбаланс классов:** мошенничество составляет всего 0.17% всех транзакций.

## Используемые технологии
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- RandomForestClassifier с балансировкой классов (class_weight='balanced')

## Результаты
| Метрика | Значение |
|---------|----------|
| F1-Score (мошенничество) | **0.85** |
| Precision (мошенничество) | 0.89 |
| Recall (мошенничество) | 0.82 |

## Как запустить
1. Установи зависимости: `pip install -r requirements.txt`
2. Открой ноутбук: `jupyter notebook fraud_detection.ipynb`
3. Запусти все ячейки

## Файлы
- `fraud_detection.ipynb` — основной ноутбук
- `fraud_detector_model.pkl` — сохранённая модель
- `src/` — вспомогательный код
